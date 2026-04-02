# GitHub SSH 配置 SOP (WSL/Linux)

## 前置条件
- 已安装 Git
- 已安装 OpenSSH

---

## 步骤 1: 生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- 按回车接受默认路径 `~/.ssh/id_ed25519`
- 提示输入密码时直接回车（不设置密码）

## 步骤 2: 查看公钥并添加到 GitHub

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的公钥，登录 GitHub → Settings → SSH and GPG keys → New SSH key，粘贴公钥。

## 步骤 3: 配置 Git 用户信息

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

## 步骤 4: 添加私钥到 SSH Agent

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 步骤 5: 配置 SSH（端口 443 方式）

如果直接连接 22 端口失败（如 Connection closed），使用此配置：

```bash
cat > ~/.ssh/config << 'EOF'
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
EOF
```

## 步骤 6: 添加 GitHub Host Key

```bash
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
```

或手动添加（推荐）：

```bash
echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" > ~/.ssh/known_hosts
```

## 步骤 7: 测试连接

```bash
ssh -T git@github.com
```

成功响应：
```
Hi [Username]! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## 附录：常见问题

### 问题：22 端口被封锁
**解决**：使用端口 443，如步骤 5 配置。

### 问题：Host key verification failed
**解决**：确保 known_hosts 包含 GitHub 的 host key，见步骤 6。

### 问题：需要代理
如果代理在宿主机（Windows），WSL 中配置：
```bash
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    ProxyCommand nc -X 5 -x 127.0.0.1:7897 %h %p
EOF
```
（端口 7897 需与代理软件配置一致）

---

## 权限要求

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```
