/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  // domain/ 是纯 TS，测试跑在 node 环境里，不需要 DOM。
  test: {
    environment: 'node',
    include: ['tests/**/*.spec.ts'],
  },
});
