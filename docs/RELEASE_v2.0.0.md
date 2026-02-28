# v2.0.0 Release

**Date:** Feb 28, 2026  
**Grade:** A → A+ roadmap  
**Status:** Development phase complete, ready for pre-training

## What's New

### ✨ UI Overhaul
- Flat design, zero gradients, portfolio-aligned typography
- Dark/light theme toggle with persistent preference
- Responsive mobile layout
- Clean message bubbles, improved readability

### 🏗️ Project Restructure
- Organized codebase: `src/`, `scripts/`, `docs/`, `data/reports/`
- Moved loose files out of root (7 files → docs/)
- Better hierarchy for maintenance and discovery

### 📖 Documentation
- Comprehensive README with architecture diagram (SVG)
- Project structure documented
- Performance metrics & training status visible

### 🎯 Model Status
- v2.0 architecture: 65M params, 8K context, Flash Attention
- BPE tokenizer implemented
- Balanced dataset: 7K examples (Phase 1 ✓, Phase 2 ✓)
- Training script ready (Phase 3 → waiting for training window)
- ONNX export pipeline prepared (Phase 4 planned)

### 🚀 Deployment
- Live at arthur-prod.vercel.app
- API fallback functional
- Vercel CI/CD wired

## A+ Roadmap (Next 6-8 weeks)

| Phase | Task | Duration | Grade Impact |
|-------|------|----------|--------------|
| 1 | v2.0 pre-training (65M params) | 2 weeks | A → A |
| 2 | Type hints + test coverage | 1 week | A → A+ |
| 3 | GitHub Actions CI/CD | 1 week | A+ |
| 4 | Docs & Jupyter notebooks | 1 week | A+ |
| 5 | Production hardening | 2 weeks | A+ |

## Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | ~65% | 85%+ |
| Type Hint Coverage | 20% | 100% |
| Math Benchmark | 78% | 80%+ |
| Inference Latency | ~50ms/token | <40ms |
| Documentation | 70% | 100% |

## Known Issues

- v2.0 pre-training not yet started (waiting for idle compute)
- Math benchmark still at research-prototype level
- CI/CD minimal (.github/workflows sparse)

## Contributors

- Joshua Trommel (nulljosh)
- Samantha (codebase reorganization, UI refresh)

---

**Next Steps:**
1. Monitor cron daemon for v2.0 training kickoff
2. Review performance benchmarks post-training
3. Iterate on math accuracy
4. Release v2.0 final with A+ certification

