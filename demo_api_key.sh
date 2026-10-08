#!/usr/bin/env bash
# Makes a new API key, saves it as the GitHub secret API_KEY (if gh is set up), and starts the API with it.
set -euo pipefail
cd "$(dirname "$0")"

export API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh secret set API_KEY --body "$API_KEY" && echo "GitHub secret API_KEY updated."
else
  echo "gh not available or not logged in: add API_KEY as a GitHub secret by hand."
fi

echo
echo "API key (paste into Swagger > Authorize):"
echo "$API_KEY"
echo
echo "Swagger: http://127.0.0.1:8000/docs"
exec python -m uvicorn app:app --port 8000
