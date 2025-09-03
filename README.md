This was/is an experiment to see if I could get LLMs to write a book, which is too large to fit in the context window, where I would act something like an editor or advisor rather than a writer.
I haven't paid much attention to how accurate or well written it is.
I'd advise against trying to learn from it just yet.

The strategy was a version of divide and conquer.
We wrote a preface and a detailed table of contents.
Based on that we wrote a Chapter.Section level dependency graph and used that to decide what needed to be in context when drafting a new section.
When starting a new section, we would work backwards through the graph and insert only the required sections into context.
This did require organizing the book in a way where this never ended up pulling in too much.
