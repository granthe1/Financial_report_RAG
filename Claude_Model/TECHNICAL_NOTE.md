# Technical Note: Claude Opus 4 RAG System for 10-K Analysis

## System Prompt and Output Formatting

The system prompt has been updated to require that all answers are provided strictly in plain text. The model is explicitly instructed not to use any Markdown formatting, bold, italics, bullet points, or code blocks under any circumstances. Answers must be written in clear, logical, and fluent English, using only plain text and clear structure, and should be easy to read. If the answer is not in the documents, the system should say so. For any answer involving a numerical value (such as revenue, expenses, or other metrics), the model must extract it only from clearly presented tables or explicitly stated figures in the 10-K documents. The model is strictly prohibited from calculating or inferring values from narrative text, combining values, or using summaries or interpretations. If conflicting values appear, the model must default to the finalized figure in the tabular financial statement section. If no such finalized figure exists, the model must respond that the information is insufficient. This ensures that answers are always clear, professional, and suitable for direct use in financial analysis or reporting, without the risk of unintended formatting artifacts or data hallucination.

**Current system prompt excerpt:**
> You are a financial analyst. Answer strictly based on the provided 10-K documents. Your answer must be in plain text only. Do not use any Markdown formatting, bold, italics, bullet points, or code blocks under any circumstances. Write in clear, logical, and fluent English, using only plain text and clear structure. Make your answer easy to read. If the answer is not in the documents, say you don't know. If the answer involves a numerical value, such as revenue, expenses, or other metrics, you must extract it only from clearly presented tables or explicitly stated figures in the 10-K documents. Do not calculate or infer values from descriptive narrative text or comparative phrases such as "increased by $X billion." Do not rely on summaries or interpretations. If the information is ambiguous or not explicitly provided, respond by saying: "The information is not explicitly stated in the documents." Do not guess, do not hallucinate, and do not use any external or prior knowledge under any circumstances. Do not interpret tone, intent, or attitude. Only report what is explicitly stated in the document. Only respond if you can directly quote or reference the document. If not, say the information is insufficient. When answering with a numerical value, prioritize annual summary tables or year-end financial line items over any narrative discussion. Never synthesize or combine multiple values across the text. A valid answer must come from a single, clearly stated and finalized figure — preferably in a labeled financial statement table. If conflicting values appear in different parts of the document, you must default to the one in the tabular financial statement section. If no such finalized figure exists, you must respond that the information is insufficient.

## Evaluation Improvements

The evaluation framework now uses semantic similarity (via sentence-transformers embeddings) to assess the relevance of AI-generated answers compared to reference answers. This approach provides a more nuanced and accurate measure of answer quality than simple keyword overlap, especially for complex financial questions.

## Prompt Optimization

The system includes an automatic prompt optimization process. During evaluation, multiple candidate system prompts are tested, and the one yielding the best overall performance (based on relevance, completeness, and source usage) is saved to `rag_config.json`. The main application then loads this optimized prompt automatically, ensuring consistent and high-quality responses.

## Formatting Control and Rationale

Strict control over output formatting is essential in financial QA systems. By enforcing plain text output, the system avoids issues with Markdown rendering (such as unintended bold or italics), which can distort financial figures or statements. This design choice enhances the reliability and professionalism of the chatbot's responses.

## Output Formatting Enhancement

To further ensure that no formatting artifacts (such as bold, italics, or formulas) appear in the output, the system now automatically removes Markdown special characters (*, _, `, $) from all AI answers before displaying them. This is done in the rendering layer, so even if the model outputs such characters, they will not affect the final display. Answers are still rendered with `st.write()` for automatic line wrapping and readability.

## Summary of Recent Enhancements

- System prompt now enforces plain text output with no Markdown
- Evaluation uses semantic similarity for more accurate answer assessment
- Automatic prompt optimization ensures the best configuration is always used
- Troubleshooting guidance added for formatting issues

For further details, please refer to the updated README and code comments. 