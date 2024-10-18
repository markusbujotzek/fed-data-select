# fed-data-select

This repo was set up in the following way:

'''

# Clone your fork of FLamby (if forking)
git clone https://github.com/YOUR_USERNAME/FLamby.git
cd FLamby

# Create a new branch
git checkout -b your-feature-branch

# Add FLamby as a submodule (if creating a new project)
git clone https://github.com/YOUR_USERNAME/your-project.git
cd your-project
git submodule add https://github.com/owkin/FLamby.git

# Add your new code
git add .
git commit -m "Initial commit with FLamby and new code"

# Push to GitHub
git push origin your-feature-branch

# Update FLamby (if needed)
git submodule update --remote --merge
git add FLamby
git commit -m "Update FLamby submodule"
git push origin your-feature-branch

'''
