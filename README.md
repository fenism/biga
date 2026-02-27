# A-Share Stock Analysis App

基于 Streamlit 的 A 股全市场选股与量化分析工具。

## 特色功能

- **宏观大盘分析**：集成流动性、情绪、趋势多维度分析。
- **量化选股**：集成强势股进攻与弱势股抄底双重策略。
- **个股诊断**：利用 Gemini AI 进行深度个股 SOP 诊断（需配置 API Key）。
- **交易管理**：模拟持仓与自选股管理。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载行情数据

在应用侧边栏点击「立即下载行情数据」，数据将保存至本地 `data/market_data/` 目录。

### 3. 配置 Gemini API Key

本应用使用 Google Gemini 进行 AI 分析。请在 `.streamlit/secrets.toml` 中配置（本地开发）或在 Streamlit Cloud 环境变量中设置：

```toml
GEMINI_API_KEY = "你的_API_KEY"
```

### 4. 运行应用

```bash
streamlit run app.py
```

## 注意事项

- 行情数据存储在本地，不随代码库分发，请首次使用时务必点击下载。
- `.gitignore` 已配置排除本地数据及个人交易记录。
