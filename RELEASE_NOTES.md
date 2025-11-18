# Release v0.1.0 - Draft

Summary
- Initial project boilerplate for Music Feature Analysis
- FastAPI inference skeleton and admin endpoint
- Model loader abstraction and sample model script
- Dockerfile and GitHub Actions for CI and Docker publish
- Tests, mypy, and ruff configured

Publishing steps (once repository has commits and CI secrets configured):

1. Ensure all changes are committed and pushed to `main`.
2. Create an annotated tag locally:

   git tag -a v0.1.0 -m "v0.1.0 - Initial boilerplate and CI/Docker scaffolding"

3. Push tag to remote:

   git push origin v0.1.0

4. CI will pick up the tag and run the release workflow to build/publish artifacts.

Notes
- If you want to publish a Docker image manually, use `docker build` (requires Docker Desktop) and push to your registry.
