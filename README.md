# **AI Powered Data Analysis System**
AI-powered business analytics tool that converts natural language queries into Pandas code, executes them in a Docker sandbox, and returns insights with recommendations.

# **Features**
* Natural language querying over structured CSV datasets
* Automatic preprocessing and schema normalization
* LLM-generated Pandas code execution
* Controlled Docker sandbox execution
* Structured outputs:
    * Answer
    * Analysis
    * Recommendation
* Multi-model support:
    * GPT-4o Mini
    * Claude 4.5 Sonnet
    * Gemini 2.5 Flash
    * LLaMA 3.3 70B Instruct
* Cloud deployment using AWS EC2 and Vercel
* Quantitative and qualitative LLM evaluation pipeline

# **System Architecture**
The system follows a full-stack architecture consisting of:
* Frontend: React (Vite)
* Backend: FastAPI
* Execution Environment: Docker sandbox
* Deployment:
    * Frontend hosted on Vercel
    * Backend hosted on AWS EC2 with systemd
* LLM Provider: OpenRouter API

# **Pipeline**
User Query → Prompt Construction → LLM Code Generation → Sandbox Execution → Result Structuring → Recommendations

# **Live Demo**
Project Website: https://a-insight-project.vercel.app/

# **Evaluation**
The system was evaluated using ~100 analytical queries across:
* Aggregation
* Comparison
* Filtering
* Ranking
* Time-series analysis
* Edge-case scenarios

# **License** 
GNU General Public License v3.0 (GPL-3.0)
