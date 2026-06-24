Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must contain a raw JSON array of integers enclosed within a `<ranking>` and `</ranking>` XML tag.
Do not wrap the JSON in Markdown backticks. Do not include any introductory or explanatory text outside the XML tags.

For example:
<ranking>[75, 12, 34, 2, 1]</ranking>

Ranking:
