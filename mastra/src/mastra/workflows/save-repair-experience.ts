import { constants } from "node:fs";
import {
  access,
  mkdir,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

import { createStep, createWorkflow } from "@mastra/core/workflows";
import { z } from "zod";

import { ragConfig } from "../rag/config.js";
import { ingestMarkdownDocument } from "../rag/ingestion.js";

const unsafeActionPattern = /(?:带电操作|短接(?:保险|保护)|拆(?:修|开).{0,8}(?:高压|电容)|徒手.{0,8}高压)/u;

const experienceInputSchema = z.object({
  experienceId: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/),
  applianceType: z.enum(["refrigerator", "television", "monitor"]),
  symptom: z.string().trim().min(2).max(2_000),
  diagnosis: z.string().trim().min(2).max(4_000),
  actions: z
    .string()
    .trim()
    .min(2)
    .max(6_000)
    .refine((value) => !unsafeActionPattern.test(value), "操作步骤包含危险的带电或高压维修内容"),
  result: z.string().trim().min(2).max(2_000),
  safetyNotes: z.string().trim().min(2).max(2_000),
});

const reviewSchema = z.object({
  approved: z.boolean(),
  reviewer: z.string().trim().min(1).max(100).optional(),
});

const workflowOutputSchema = z.object({
  status: z.enum(["approved", "rejected"]),
  experienceId: z.string(),
  source: z.string(),
  indexedChunks: z.number().int().nonnegative(),
});

export type RepairExperienceInput = z.infer<typeof experienceInputSchema>;

function sanitizePersonalData(value: string): string {
  return value
    .replace(/\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/g, "[邮箱已脱敏]")
    .replace(/(?<!\d)1[3-9]\d{9}(?!\d)/g, "[手机号已脱敏]")
    .replace(/(?<!\d)\d{17}[\dXx](?!\d)/g, "[证件号已脱敏]")
    .replaceAll("\0", "")
    .trim();
}

function markdownValue(value: string): string {
  return sanitizePersonalData(value)
    .split(/\r?\n/)
    .map((line) => (/^(?:---|#{1,6}\s)/.test(line) ? `\\${line}` : line))
    .join("\n");
}

export function renderRepairExperienceMarkdown(input: RepairExperienceInput): string {
  return `---
document_id: "${input.experienceId}"
title: "${input.experienceId} 维修经验"
knowledge_scope: "repair"
appliance_type: "${input.applianceType}"
document_version: "1"
review_status: "draft"
---

# ${input.experienceId} 维修经验

## 故障现象

${markdownValue(input.symptom)}

## 诊断过程

${markdownValue(input.diagnosis)}

## 处理步骤

${markdownValue(input.actions)}

## 处理结果

${markdownValue(input.result)}

## 安全提示

${markdownValue(input.safetyNotes)}
`;
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function saveExperienceDraft(
  input: RepairExperienceInput,
  root = ragConfig.experienceDirectory,
): Promise<{ draftPath: string; markdown: string }> {
  const parsed = experienceInputSchema.parse(input);
  const draftDirectory = path.join(root, "drafts");
  await mkdir(draftDirectory, { recursive: true });
  const draftPath = path.join(draftDirectory, `${parsed.experienceId}.md`);
  const markdown = renderRepairExperienceMarkdown(parsed);
  try {
    await writeFile(draftPath, markdown, { encoding: "utf8", flag: "wx" });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    const current = await readFile(draftPath, "utf8");
    if (current !== markdown) {
      throw new Error(`经验 ID ${parsed.experienceId} 已存在且内容不同。`);
    }
  }
  return { draftPath, markdown };
}

export async function approveExperienceDraft(
  experienceId: string,
  root = ragConfig.experienceDirectory,
  reviewer = "manual-review",
): Promise<string> {
  const safeId = z
    .string()
    .regex(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/)
    .parse(experienceId);
  const draftPath = path.join(root, "drafts", `${safeId}.md`);
  const approvedDirectory = path.join(root, "approved");
  const approvedPath = path.join(approvedDirectory, `${safeId}.md`);
  await mkdir(approvedDirectory, { recursive: true });
  const reviewedMarkdown = (markdown: string) =>
    markdown.replace(
      'review_status: "draft"',
      `review_status: "approved"\napproved_by: ${JSON.stringify(
        sanitizePersonalData(reviewer) || "manual-review",
      )}`,
    );
  if (await fileExists(approvedPath)) {
    if (!(await fileExists(draftPath))) return approvedPath;
    const [draft, approved] = await Promise.all([
      readFile(draftPath, "utf8"),
      readFile(approvedPath, "utf8"),
    ]);
    if (reviewedMarkdown(draft) !== approved) {
      throw new Error(`经验 ID ${safeId} 已批准，不能用不同内容覆盖。`);
    }
    await unlink(draftPath);
    return approvedPath;
  }
  const draft = await readFile(draftPath, "utf8");
  await writeFile(draftPath, reviewedMarkdown(draft), "utf8");
  await rename(draftPath, approvedPath);
  return approvedPath;
}

const reviewExperienceStep = createStep({
  id: "review-repair-experience",
  inputSchema: experienceInputSchema,
  outputSchema: workflowOutputSchema,
  resumeSchema: reviewSchema,
  suspendSchema: z.object({
    experienceId: z.string(),
    markdown: z.string(),
    message: z.string(),
  }),
  execute: async ({ inputData, resumeData, suspend }) => {
    const { markdown } = await saveExperienceDraft(inputData);
    if (!resumeData) {
      return suspend({
        experienceId: inputData.experienceId,
        markdown,
        message: "请审核该维修经验；批准后才会进入正式向量知识库。",
      });
    }
    if (!resumeData.approved) {
      return {
        status: "rejected" as const,
        experienceId: inputData.experienceId,
        source: `drafts/${inputData.experienceId}.md`,
        indexedChunks: 0,
      };
    }
    const approvedPath = await approveExperienceDraft(
      inputData.experienceId,
      ragConfig.experienceDirectory,
      resumeData.reviewer,
    );
    const indexed = await ingestMarkdownDocument(approvedPath, {
      sourceRoot: ragConfig.experienceDirectory,
      knowledgeScope: "repair",
    });
    return {
      status: "approved" as const,
      experienceId: inputData.experienceId,
      source: `approved/${inputData.experienceId}.md`,
      indexedChunks: indexed.chunks,
    };
  },
});

export const saveRepairExperienceWorkflow = createWorkflow({
  id: "save-repair-experience",
  description: "保存、人工审核并在批准后向量化维修经验。",
  inputSchema: experienceInputSchema,
  outputSchema: workflowOutputSchema,
})
  .then(reviewExperienceStep)
  .commit();
