@echo off
cd github_upload

git init
git add .
git commit -m "Initial commit: APO-MORL implementation and experimental results

- Complete MORL algorithm implementation
- Comprehensive experimental validation
- Publication-quality figures in multiple formats  
- Video demonstration of UR5 manufacturing system
- Statistical analysis with significance testing
- Full reproducibility package"

git branch -M main
git remote add origin https://github.com/ClaudioUrrea/ur5_CoppeliaSim_EDU.git
git push -u origin main

echo "Repository uploaded to GitHub successfully!"
pause