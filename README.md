### The Problem

Individuals and enterprises both accumulate knowledge — personal notes, internal project context — that becomes hard to search as it grows. This application is an AI-powered knowledge base assistant: the user supplies a folder of plain files (`.txt`/`.json` for now) as the knowledge base, then asks questions in natural language. The assistant answers strictly from that content — never from general knowledge or the internet — and says so explicitly when an answer isn't present, rather than guessing. This matters most for private knowledge with no public source to fall back on, where an ungrounded answer would be misleading rather than just unhelpful.

The application is a Python terminal application with no graphical front end. Adding a file to the `knowledge/` folder is the upload mechanism; the terminal is the interface.

### Use Cases

1. **Grounded knowledge base Q&A.** The user asks a question and receives an answer sourced only from their own files, or an explicit "not found" if it isn't covered. The knowledge base's subject matter can be anything — a student's study notes for exam review, or a company's internal project documentation for onboarding. The mechanism and test cases are the same either way.
2. **Grounded artifact generation.** The user asks the assistant to produce a written artifact based on the knowledge base; the result is saved to the `output/` folder for the user to consume.

### AI Tooling Used to Refine This Document

Claude helped turn early drafts of the writings into clearer, more concise, and grammatically correct language.
