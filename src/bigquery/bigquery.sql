WITH
  commits AS (
  SELECT
    author.email,
    LOWER(REGEXP_EXTRACT(diff.new_path, r'\.([^\./\(~_ \- #]*)$')) lang,
    diff.new_path AS path,
    author.date
  FROM
    `bigquery-public-data.github_repos.commits`,
    UNNEST(difference) diff
  WHERE
    EXTRACT(YEAR
    FROM
      author.date)=2016 )
SELECT
  lang,
  COUNT(path) AS numcommits
FROM
  commits
WHERE
  LENGTH(lang) < 8
  AND lang IS NOT NULL
  AND REGEXP_CONTAINS(lang, '[a-zA-Z]')
GROUP BY
  lang
HAVING
  numcommits > 100
ORDER BY
  numcommits DESC