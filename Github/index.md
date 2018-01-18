# Github: The Basics

Think of Github as the Facebook for those who write code. You want to share your code _(create a repository)_, you want to like someone's code _(starring repositories)_, you want to jump into conversations about code because they might have a bug _(to submit an issue)_. Your future employer might also ask for your Github, so building your Github presence is like building a portfolio. But at its core is Git, which is a version control system. In a team setting, you will be able to independently and parallely work on a code and later on _merge_ the changes into a single file. These parallel workflows are called _branches_. I will walk you through some of the concepts and techniques that lay the foundation for you to master Github. We will be simulating a coding project where you are supposed to overcome the challenges that come with versioning.

## Exercise 1: Working on your own

1. Create a Folder which will be your parent directory for your code.
2. Create a text file in which the first line reads as "first line"
3. Open Github and create a repository. Save the link it provides you.
4. Open Terminal type ```cd``` and drag your folder to the Terminal. Then type the following commands:

**Initializing the repository**

```bash
git init
```

**Adding files**

Adding all files
```bash
git add .
```

Adding a specific file
```bash
git add
```
and drag your file to the terminal

**Committing the changes**

Alternatively, here you may replace ```"first commit"``` with another commit message. It helps you to see what were the last updates you made to the files.

```bash
git commit -m"first commit"
```

**Adding a Remote**

```bash
git add remote origin
``` 

paste here the URL that you saved

**Staging (pushing) the changes**

```bash
git push -u origin master
```


