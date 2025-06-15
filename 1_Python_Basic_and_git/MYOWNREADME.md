---

````markdown
## 🔧 Git & Version Control Journey — Day 1 Summary

This section documents the version control setup process and errors I faced while starting the *Machine Learning in Production* course.

---

### ✅ Step 1: Cloned Instructor's Repository

I ran:

```bash
git clone https://github.com/tharhtetsan/ML-in-Prod-batch-2.git
````

This cloned the original repo, but I couldn’t push because I didn’t have write access to the instructor’s repository.

---

### ❌ Error: Push Denied (403)

```bash
remote: Permission to tharhtetsan/ML-in-Prod-batch-2.git denied to robinglory.
fatal: unable to access 'https://github.com/...': The requested URL returned error: 403
```

**Why?**
I cloned a repo I don’t own. Only the owner (instructor) can push changes to it.

---

### ✅ Step 2: Forked the Repository on GitHub

To make it my own, I:

1. Went to GitHub and forked the instructor’s repo to `robinglory/ML-in-Prod-batch-2`.
2. Updated the origin URL locally:

```bash
git remote set-url origin git@github.com:robinglory/ML-in-Prod-batch-2.git
```

---

### ✅ Step 3: Set Up SSH Authentication

#### 🔑 SSH Key Generated:

```bash
ssh-keygen -t rsa -f ~/.ssh/tharhtetsan -C "my_github_email@example.com"
```

This created:

* `tharhtetsan` → the private key
* `tharhtetsan.pub` → the public key

#### 🗝 Added SSH Key to GitHub:

1. Copied the content of `tharhtetsan.pub`
2. Went to **GitHub → Settings → SSH and GPG Keys**
3. Clicked **New SSH Key**, pasted it, and saved

#### 🧪 Verified:

```bash
ssh -T git@github.com
```

Got this response:

```bash
Hi robinglory! You've successfully authenticated, but GitHub does not provide shell access.
```

---

### ❌ Error: SSH Permission Denied in Anaconda Prompt

```bash
Permission denied (publickey).
```

**Why?**
Anaconda Prompt didn’t know how to find my SSH key. I tried to use `ssh-add`, but it failed because the authentication agent wasn't running.

```bash
Could not open a connection to your authentication agent.
```

---

### ✅ Final Decision

I decided to use **Git Bash** for all version control tasks because:

* It supports SSH natively
* `ssh-agent` and `ssh-add` work correctly
* Git push & pull work smoothly after authentication

---

### 💡 Conclusion

* Always fork a repo if I need to make changes and push
* Use Git Bash for git-related work (push, pull, clone, commit)
* Keep SSH keys safe and linked to GitHub
* Anaconda Prompt is better used only for Python environment setup

---

✅ Everything is now working perfectly in Git Bash with my SSH key. I can commit, push, and pull using:

```bash
git push origin main
```

