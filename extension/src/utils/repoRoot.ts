/** 项目根探测的纯逻辑（可单元测试）：从任意文件路径向上找"仓库根"。 */

/** 判断目录是否像 tf2-tutorial 仓库根（特征：chapters/ 或 手册/requirements 存在） */
export type RepoRootPredicate = (dir: string) => boolean;

/**
 * 从 startFile（或目录）向上最多 maxUp 层，找第一个满足 isRoot 的目录。
 * 找不到返回 undefined。
 */
export function findRepoRoot(
  startPath: string,
  isRoot: RepoRootPredicate,
  maxUp: number = 6,
): string | undefined {
  let dir = startPath;
  for (let i = 0; i <= maxUp; i++) {
    if (isRoot(dir)) {
      return dir;
    }
    const parent = dirnameOf(dir);
    if (parent === dir) {
      break; // 到达盘符根
    }
    dir = parent;
  }
  return undefined;
}

// 避免在纯模块里引 Node：由调用方注入 dirname
function dirnameOf(p: string): string {
  const norm = p.replace(/[\\/]+$/, "");
  const idx = Math.max(norm.lastIndexOf("\\"), norm.lastIndexOf("/"));
  return idx <= 0 ? norm : norm.slice(0, idx);
}
