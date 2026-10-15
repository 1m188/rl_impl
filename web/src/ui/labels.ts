/** 收尾原因的中文标签（两个游戏共用；未知原因原样显示） */
export const END_REASON_LABEL: Record<string, string> = {
  goal: '到终点',
  trap: '踩陷阱',
  death: '撞死',
  timeout: '超时',
  full: '通关',
};

export function endReasonLabel(reason: string): string {
  return END_REASON_LABEL[reason] ?? reason;
}
