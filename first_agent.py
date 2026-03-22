import os
import urllib.request
import subprocess
import schedule
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = "dummy-not-used"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = "Regomoditswe-code"
REPO_NAME = "ds-agents"
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
WORK_DIR = r"C:\Users\mphel\ds-agents"

from crewai import Agent, Task, Crew, LLM

llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# ── Download datasets ──────────────────────────────────────────────
datasets = {
    "spotify.csv":  "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv",
    "finance.csv":  "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv",
    "nba.csv":      "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv"
}
for filename, url in datasets.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

# ── Agents ─────────────────────────────────────────────────────────
ds_agent = Agent(
    role="Data Scientist",
    goal="Analyze datasets and extract key insights",
    backstory="Expert data scientist in music, finance, and sports analytics",
    llm=llm, verbose=True
)
viz_agent = Agent(
    role="Visualization Expert",
    goal="Suggest the best charts to represent data insights",
    backstory="Expert in data visualization who knows which charts communicate data best",
    llm=llm, verbose=True
)
report_agent = Agent(
    role="Report Writer",
    goal="Write a clear professional README.md from the analysis and visualizations",
    backstory="Technical writer who turns data insights into polished markdown reports",
    llm=llm, verbose=True
)
github_agent = Agent(
    role="GitHub Manager",
    goal="Commit and push all project files to GitHub",
    backstory="DevOps expert who manages git repositories and version control",
    llm=llm, verbose=True
)
linkedin_agent = Agent(
    role="LinkedIn Content Creator",
    goal="Write an engaging LinkedIn post about the data analysis findings",
    backstory="Professional content creator who writes compelling data science posts for LinkedIn",
    llm=llm, verbose=True
)
business_agent = Agent(
    role="Business Analyst",
    goal="Turn data insights into clear actionable business recommendations",
    backstory="Senior business analyst who bridges data science and real-world strategy",
    llm=llm, verbose=True
)

# ── Tasks ──────────────────────────────────────────────────────────
analyze_task = Task(
    description="""Analyze all three datasets:
    1. spotify.csv - music streaming (genres, tempo, popularity, danceability)
    2. finance.csv - S&P 500 financials (sector, price, earnings, market cap)
    3. nba.csv - NBA player RAPTOR scores (performance, age, wins)
    For each: key stats, top findings, interesting correlations.""",
    expected_output="Detailed analysis of all 3 datasets with key insights",
    agent=ds_agent
)

viz_task = Task(
    description="""Based on the analysis, recommend 3-4 specific charts per dataset.
    For each chart specify: chart type, columns for each axis, why it works, color coding.""",
    expected_output="Specific visualization plan for all 3 datasets",
    agent=viz_agent,
    context=[analyze_task]
)

business_task = Task(
    description="""Based on the data analysis, provide actionable business recommendations:
    - Spotify: what makes a song go viral? What should artists focus on?
    - Finance: which sectors look strongest? Any red flags?
    - NBA: what player traits predict team success?
    Be specific and practical.""",
    expected_output="Actionable business recommendations for each dataset",
    agent=business_agent,
    context=[analyze_task]
)

report_task = Task(
    description="""Write a professional README.md that includes:
    - Project title and description
    - Dataset summaries
    - Key findings from the analysis
    - Business recommendations
    - Visualization descriptions
    - How to run the project
    Format it beautifully with markdown headers, bullet points, and emojis.""",
    expected_output="Complete README.md content as a markdown string",
    agent=report_agent,
    context=[analyze_task, viz_task, business_task]
)

github_task = Task(
    description=f"""Your job is to provide the exact git commands needed to:
    1. Initialize the repo if needed
    2. Add all files
    3. Commit with today's date
    4. Push to: {REPO_URL}
    List each command clearly on its own line.""",
    expected_output="List of git commands to run",
    agent=github_agent,
    context=[report_task]
)

linkedin_task = Task(
    description="""Write an engaging LinkedIn post (max 300 words) about today's analysis. Include:
    - A hook opening line
    - 2-3 surprising insights from the data
    - What this means for the industry
    - A call to action to check the GitHub repo
    - Relevant hashtags (#DataScience #AI #Python #MachineLearning)
    Make it sound human, not robotic.""",
    expected_output="A ready-to-post LinkedIn update",
    agent=linkedin_agent,
    context=[analyze_task, business_task]
)

# ── Run pipeline ───────────────────────────────────────────────────
def run_pipeline():
    print("\n🚀 Starting daily pipeline...\n")

    crew = Crew(
        agents=[ds_agent, viz_agent, business_agent, report_agent, github_agent, linkedin_agent],
        tasks=[analyze_task, viz_task, business_task, report_task, github_task, linkedin_task]
    )
    result = crew.kickoff()

    # Generate charts
    print("\n📊 Generating charts...")
    os.makedirs("charts", exist_ok=True)

    try:
        df = pd.read_csv("spotify.csv")
        plt.figure(figsize=(10, 5))
        top_genres = df['playlist_genre'].value_counts().head(8)
        sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis")
        plt.title("Top Spotify Genres by Track Count")
        plt.tight_layout()
        plt.savefig("charts/spotify_genres.png", dpi=150)
        plt.close()
        print("  ✅ spotify_genres.png")
    except Exception as e:
        print(f"  ⚠️ Spotify chart error: {e}")

    try:
        df = pd.read_csv("finance.csv")
        plt.figure(figsize=(10, 5))
        sector_counts = df['Sector'].value_counts()
        sns.barplot(x=sector_counts.values, y=sector_counts.index, palette="coolwarm")
        plt.title("S&P 500 Companies by Sector")
        plt.tight_layout()
        plt.savefig("charts/finance_sectors.png", dpi=150)
        plt.close()
        print("  ✅ finance_sectors.png")
    except Exception as e:
        print(f"  ⚠️ Finance chart error: {e}")

    try:
        df = pd.read_csv("nba.csv")
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=df, x="age", y="raptor_total", alpha=0.5, color="steelblue")
        plt.title("NBA Player Age vs RAPTOR Score")
        plt.tight_layout()
        plt.savefig("charts/nba_raptor.png", dpi=150)
        plt.close()
        print("  ✅ nba_raptor.png")
    except Exception as e:
        print(f"  ⚠️ NBA chart error: {e}")

    # Save README
    print("\n📝 Saving README.md...")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(str(result))
    print("  ✅ README.md saved")

    # Save LinkedIn post
    print("\n💼 Saving LinkedIn draft...")
    with open("linkedin_post.txt", "w", encoding="utf-8") as f:
        f.write(str(linkedin_task.output) if linkedin_task.output else "LinkedIn post not generated")
    print("  ✅ linkedin_post.txt saved — review before posting!")

    # Push to GitHub
    print("\n🐙 Pushing to GitHub...")
    try:
        os.chdir(WORK_DIR)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "remote", "add", "origin", REPO_URL], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"Daily analysis update"], check=True)
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("  ✅ Pushed to GitHub!")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ Git error: {e}")

    print("\n✅ Pipeline complete!")

# ── Scheduler ─────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()  # Run immediately on start
    schedule.every().day.at("08:00").do(run_pipeline)
    print("\n⏰ Scheduler running — pipeline will repeat daily at 08:00")
    while True:
        schedule.run_pending()
        time.sleep(60)