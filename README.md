# Large Scale Uncertainty 
```
conda create -n lscaleuq python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


# How to contribute? 
```
- Work on a specific branch
- Make a pull request on github: Do it as soon as you start working on your new branch, put [WIP] in the title of your pull request as long as you are not done with your modifications. Once you are done, ensure that your code is linted and runs without bugs.
- After approved, you can merged your branch and delete it on github. Next, do the following steps on your local machine:
    - git checkout main
    - git pull
    - git branch -D your_branch
    - git checkout -b your_new_branch
    - git push --set-upstream origin your_new_branch

- It can happen that your working branch is not up to date with the remote main due to other people merging their own branches. If so you must merge the remote main into your local branch. To do so, follow the following steps:
    - git checkout main
    - git pull
    - git checkout your_branch
    - git merge main
    - Manually resolve the conflicts if needed
    - git commit -am "merging confits resolved"
    - git push
```