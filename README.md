# DS-Agents: Autonomous Multi-Agent Data Science Pipeline

> Built with Python, CrewAI & Groq — analyzes datasets, generates visualizations, writes reports, and pushes to GitHub automatically. No human required.

---

## What This Does

This project runs **6 AI agents in sequence**, each with a specific role:

| Agent | Role |
|-------|------|
| Data Scientist | Analyzes datasets and extracts key insights |
| Visualization Expert | Recommends the best charts for the data |
| Chart Generator | Produces PNG charts using matplotlib & seaborn |
| Report Writer | Writes a professional README from the findings |
| GitHub Manager | Commits and pushes everything to this repo |
| LinkedIn Agent | Drafts a LinkedIn post for review before posting |

All agents share context — each one builds on the previous agent's output.

---

## Datasets Analyzed

- **Spotify** — music streaming data (genres, tempo, popularity, danceability)
- **S&P 500 Financials** — company financials (sector, price, earnings, market cap)
- **NBA RAPTOR Scores** — player performance metrics (RAPTOR score, age, wins)

---

## Key Findings

### Music
- Tempo and danceability have a **0.55 correlation** — upbeat songs get more streams
- Pop and latin genres dominate track counts on Spotify

### Finance
- Technology companies average **$234.6B market cap** — highest of any sector
- Earnings per share is the strongest predictor of stock price

### NBA
- RAPTOR score and wins have a **0.85 correlation** — performance is measurable
- Player value peaks between ages 24–28

---

## Tech Stack

- **Python 3.11**
- **CrewAI** — multi-agent orchestration
- **Groq** — LLM inference (llama-3.3-70b-versatile)
- **Matplotlib & Seaborn** — chart generation
- **GitPython** — automated GitHub commits
- **Schedule** — daily pipeline automation
- **python-dotenv** — secure environment variables

---

## How To Run

### 1. Clone the repo
```bash
git clone https://github.com/Regomoditswe-code/ds-agents.git
cd ds-agents
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install crewai litellm matplotlib seaborn pandas schedule python-dotenv gitpython
```

### 4. Set up your .env file
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
```

### 5. Run the pipeline
```bash
python first_agent.py
```

The pipeline runs immediately then repeats **every day at 08:00** automatically.

---

## Sample Output

Charts are saved to the `/charts` folder:
- `spotify_genres.png` — top genres by track count
- `finance_sectors.png` — S&P 500 companies by sector
- `nba_raptor.png` — player age vs RAPTOR score

A `linkedin_post.txt` draft is also generated for review before posting.

---

## Author

**Regomoditswe (Potso Frans) Mphela**  
Data Science Student @ Richfield  
📍 Pretoria, South Africa  
🔗 [LinkedIn](https://www.linkedin.com/in/regomoditswe-mphela) | [GitHub](https://github.com/Regomoditswe-code)

---

## License

MIT License — free to use and build on.
