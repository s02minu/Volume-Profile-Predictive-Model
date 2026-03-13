# Volume Profile Predictive Model

Well, I was thinking about a final project for my associate degree in Data Science and Information Management course (level 5), so I thought maybe doing something I am interested in will be a smart go to and keep me hooked. The final verdict was Volume Profile.

I did some day trading as a hobby, where I encountered order flow concepts. And despite having understood the theoretical part of these concepts, I struggled when it came to applying my learning to the markets. This was in part because of the human side of the equation — my emotions. Long story short I am looking for a way to transform the decision making into a more mechanical and automatic process, hopefully helping me become more profitable. Hence this project.

I'll first be building my own version of the volume profile using my Python skills and then create a predictive model that would guide me into making the right decision. The final part would be to apply it to the market (manually), if successfully created.

---

## What is Volume Profile?

I guess it makes sense to start by explaining what Volume Profile is. Volume Profile (VP) is none other than an organization of price data in a way where we have a visual representation of where traders were willing to trade. The core idea is auction theory, which states that price will tend back to where it's considered fair value. So, if in any trading day, there was a level in which traders were willing to trade around, it's very likely that that level will be revisited and traded around soon, acting as a kind of magnet.

VP is essentially a vertical histogram mapped along the price axis, mirroring the shape of a normal distribution bell curve.

<p align="center">
<img width="700" height="3000" alt="image" src="https://github.com/user-attachments/assets/3a872d6a-1c69-471e-8b25-11724728a8d3" />

</p>

### Core Concepts

- **POC (Point of Control)** — the price level where the most volume was traded during a session. It represents the market's fairest price for that period.

- **VAH (Value Area High)** — the upper boundary of the Value Area, which is the price range that contains approximately 70% of the total traded volume during a session.

- **VAL (Value Area Low)** — the lower boundary of that same 70% volume range.

> The 70% figure is a convention (originating from CBOT Market Profile methodology), not a mathematical law. The Value Area is calculated around the POC and then expanded outward until ~70% of total volume is included.

There is more to VP than this obviously, but as this is primarily a Machine Learning project, I will not be diving deeper into VP theory. Instead, I'm going to build a model that looks at Bitcoin's volume profile and tries to predict whether price will move in a certain direction or reach a certain level. For example — price is sitting just below the POC, will it break through and accept above it, or get rejected? The model will learn from historical examples of that same setup and will generate signals if proven to have a predictive edge.

---

## Project Structure

This project is organized as a modular pipeline. Each module has a single responsibility and feeds into the next. See the flowchart for a visual overview of how everything connects.

<p align="center">
<img width="700" height="1000" alt="image" src="https://github.com/user-attachments/assets/baa1c1b9-9082-4682-8694-303a6f36f2b1" />
</p>



| Module | File | Description |
|---|---|---|
| Data Loader | `data_loader.py` | Loads raw BTCUSDT tick data from Binance into DuckDB |
| Volume Profile Engine | `volume_profile.py` | Builds daily VP and computes POC, VAH, VAL |
| Visualizer | `visualize.py` | Plots VP for sanity checking against real charts |
| Feature Engineering | `features.py` | Extracts ML features from VP levels |
| Label Generator | `labels.py` | Creates binary labels for supervised learning |
| Model | `model.py` | Trains and evaluates the ML model |
| Evaluation | `evaluation.py` | Measures predictive edge |
| Pipeline Entry Point | `main.py` | Runs the full pipeline from start to finish |

---


## data_loader.py

As the name implies, this is the step where all the BTCUSDT tick data is loaded into memory and made ready for processing.

The raw data was manually downloaded from [data.binance.vision](https://data.binance.vision) as monthly zip files covering 6 months of BTCUSDT perpetual futures trades. Each file contains every individual trade that occurred that month, including the price, quantity, timestamp, and whether the buyer or seller was the market aggressor (`is_buyer_maker`).

When attempting to load this data directly into a pandas DataFrame, the program consistently crashed due to memory exhaustion — 6 months of BTC tick data amounts to tens of millions of rows, which exceeds what 16GB of RAM can handle all at once.

<p align="center">
<img width="1598" height="707" alt="image" src="https://github.com/user-attachments/assets/1bf97894-7abd-44b2-993f-d8228da55b10" />
</p>

**The solution was DuckDB** — a database engine that runs entirely on your machine with no server setup required. Instead of loading all rows into RAM simultaneously, DuckDB processes the data file by file and only hands pandas the final query result, which is a fraction of the size.

> `data_loader.py` only needs to be run once. After that, `volume_profile.py` saves the processed data to disk and all subsequent modules load from CSV files directly.

### Data Schema

Each raw trade file contains the following columns:

| Column | Type | Description |
|---|---|---|
| `id` | bigint | Unique trade ID |
| `price` | double | Price at which the trade occurred |
| `qty` | double | Volume traded in BTC |
| `quote_qty` | double | Volume traded in USDT |
| `time` | bigint | Unix timestamp in milliseconds |
| `is_buyer_maker` | boolean | True = seller was aggressor, False = buyer was aggressor |

### Data Flow

<img width="732" height="176" alt="image" src="https://github.com/user-attachments/assets/feb34314-a423-4172-b58a-7151b19ba850" />

---

## volume_profile.py

This module is responsible for building the Volume Profile from the raw tick data loaded by `data_loader.py`. It is the core engine of the project — everything the ML model will learn from originates here.

### What it does

It takes the raw trades stored in DuckDB and aggregates them into a daily Volume Profile using $10 price buckets. For each trading day it computes three key levels — the POC, VAH, and VAL — using the 70% value area rule. It also extracts the true daily OHLC directly from raw ticks. The results are then saved to disk so that all subsequent modules can load them instantly without needing to reprocess the raw tick data.

### Why $10 buckets?

BTC trades at prices in the tens of thousands of dollars. A $1 bucket size would produce an overly granular and noisy profile, while a $100 bucket would be too coarse to be meaningful. $10 strikes a balance between precision and readability, and is a commonly used increment for BTC volume profile analysis.

### How the Value Area is calculated

Starting from the POC — the price bucket with the highest volume — the algorithm expands outward one bucket at a time, always choosing the direction with the next highest volume, until 70% of the day's total volume has been accumulated. The upper and lower boundaries of that range become the VAH and VAL respectively.

### Outputs

| File | Description |
|---|---|
| `data/df_vp.csv` | Full daily volume profile — one row per price bucket per day |
| `data/df_levels.csv` | Key VP levels per day — POC, VAH, VAL, and total volume |
| `data/df_ohlc.csv` | True daily OHLC extracted directly from raw ticks |

### Data Flow
<img width="528" height="595" alt="image" src="https://github.com/user-attachments/assets/e039eb2b-bacb-472e-862a-38cf6cb3f6ee" />


### Functions

| Function | Description |
|---|---|
| `build_daily_volume_profile(con, bucket_size)` | Queries DuckDB and returns df_vp |
| `compute_vp_levels(df_vp)` | Computes POC, VAH, VAL for each day |
| `extract_daily_ohlc(con)` | Extracts true daily OHLC from raw ticks |
| `save_to_csv(df_vp, df_levels)` | Saves dataframes to the data/ folder |
| `load_from_csv()` | Loads pre-processed data from disk for use by other modules |

---

## visualize.py

In this module the focus is the visualization of our Volume Profile in order to compare it against a real professional charting platform — in this case ATAS.

The data is loaded directly from the saved CSV files generated by `volume_profile.py` and plotted using Plotly, which provides an interactive chart where you can zoom, pan, and hover over individual price buckets to inspect exact volume values.

To validate our VP engine we plotted January 30th 2026 and compared the results against ATAS:

<p align="center">
<img width="500" height="850" alt="image" src="https://github.com/user-attachments/assets/2fc6ebf0-86cd-40ac-886b-c6bdd5e31702" /> <img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/cc83dbb9-d753-4b22-8ce5-eab51925df4d" />
</p>



| Level | Model | ATAS | Difference |
|---|---|---|---|
| POC | 82800 | 82900 | $100 |
| VAH | 83450 | 83470 | $20 |
| VAL | 81780 | 81800 | $20 |

The small differences are expected and are a direct result of the $10 price bucket aggregation. In practice these levels are close enough to be considered identical for trading purposes.

With the VP engine validated, the project is ready to move into the ML phase.

---

## features.py

This module takes the VP levels computed by `volume_profile.py` and transforms them into structured numerical inputs that the ML model can learn from. A model can't read a chart — it needs numbers, and this is where they created.

The core idea is simple — yesterday's VP describes the market context, and today's price action is the outcome. So every feature here is derived from the **previous day's** profile using pandas `shift(1)` operation, which moves each value down one row so that today's row contains yesterday's data.

### Features Built

| Feature | Description |
|---|---|
| `prev_poc` | Yesterday's Point of Control |
| `prev_vah` | Yesterday's Value Area High |
| `prev_val` | Yesterday's Value Area Low |
| `prev_va_width` | Yesterday's value area width (VAH - VAL) |
| `prev_total_volume` | Yesterday's total traded volume |
| `prev_delta` | Yesterday's buy volume minus sell volume |
| `prev_buy_ratio` | Yesterday's buy volume as a percentage of total volume |
| `prev_va_coverage` | How much of the day's range was covered by the value area |
| `prev_poc_position` | Where the POC sat within the value area (0 = bottom, 1 = top) |
| `prev_day_type` | Whether yesterday was accumulation, distribution, trending or neutral |
| `poc_direction` | Did today's POC move up or down relative to yesterday's? |
| `price_vs_prev_poc` | Is today's close above or below yesterday's POC? |
| `dist_prev_poc` | Distance from today's open to yesterday's POC |
| `dist_prev_vah` | Distance from today's open to yesterday's VAH |
| `dist_prev_val` | Distance from today's open to yesterday's VAL |

### Day Type Classification

Each day is classified based on va_coverage and delta:

- **Accumulation** — VA coverage ≥ 60% and positive delta — price ranged tightly with net buying pressure
- **Distribution** — VA coverage ≥ 60% and negative delta — price ranged tightly with net selling pressure
- **Trending** — VA coverage ≤ 40% — price moved directionally, value area is narrow relative to full range
- **Neutral** — anything in between

### Data Flow

<img width="579" height="413" alt="image" src="https://github.com/user-attachments/assets/762d9921-93fa-4a75-baf4-73e1772fc682" />


---

## labels.py

This module creates the target label that the ML model will learn to predict. If the features describe yesterday's market context, the label answers the question — **what did price actually do today?**

### The Labels

For this first iteration we focused on one primary binary question:

> **Did price accept above yesterday's VAH today?**
> - `1` = Yes — today's close was above yesterday's Value Area High
> - `0` = No — today's close remained inside or below yesterday's Value Area

This is one of the most meaningful setups in Volume Profile trading. When price closes above the previous session's VAH it signals that the market has found new value to the upside — buyers were willing to trade at higher prices and sellers did not push back hard enough to bring price back inside the value area. That's a bullish acceptance signal.

For this first iteration we also built two additional labels:

- `label_poc_bullish` — did price open above yesterday's POC and close above it too?
- `label_poc_bearish` — did price open below yesterday's POC and close below it too?

In the future there shall be other questions asked, improving the predictability of the model.

### Class Balance

After running the labels against our 6 months of data:

| Label | Yes (1) | No (0) |
|---|---|---|
| `label_vah_acceptance` | 66 (36.1%) | 117 (63.9%) |
| `label_poc_bullish` | 69 (37.7%) | 114 (62.3%) |
| `label_poc_bearish` | 52 (28.4%) | 131 (71.6%) |

### Monthly VAH Acceptance Rate

To validate the labels against real market behavior, we computed the monthly VAH acceptance rate and compared it against the actual BTCUSDT price chart for the same period:

| Month | Acceptance Rate |
|---|---|
| August 2025 | 33.3% |
| September 2025 | 43.3% |
| October 2025 | 48.4% |
| November 2025 | 26.7% |
| December 2025 | 25.8% |
| January 2026 | 38.7% |

The low acceptance rates in November and December 2025 correctly reflect the bearish market conditions visible on the chart during that period, confirming that our labels are accurately capturing real market behavior.

> A model that simply predicts "no" every time would be right 63.9% of the time — our model needs to beat that baseline to have any real edge.

### Data Flow

<img width="685" height="120" alt="image" src="https://github.com/user-attachments/assets/db749022-e0c8-4b56-81f4-196e7b6e3f08" />


---

## model.py

This module trains a Logistic Regression model on the VP features and evaluates its performance on unseen test data.

### Why Logistic Regression?

Despite the name it is actually a classification model, not a regression one. It draws a decision boundary between the two classes and outputs a probability between 0 and 1. We chose it as our baseline because it is interpretable — the coefficients directly tell us which features the model is relying on and in which direction, which is valuable for validating that the model has learned real VP concepts rather than noise.

### Training Process

The data is split 80/20 into training and test sets. Features are scaled using StandardScaler so no single feature dominates due to its magnitude. The model is trained on the 80% training set and evaluated on the 20% test set it has never seen.

### Results on Test Set

| Metric | Class 0 (No) | Class 1 (Yes) | Overall |
|---|---|---|---|
| Precision | 96% | 64% | — |
| Recall | 81% | 90% | — |
| F1 Score | 88% | 75% | — |
| Accuracy | — | — | 84% |

Baseline accuracy (majority class): 63.9%
Model accuracy: 84% — a **20.1% improvement** over baseline.

### Feature Importance

| Feature | Coefficient | Interpretation |
|---|---|---|
| `price_vs_prev_poc` | +2.44 | Strongest bullish signal — price above yesterday's POC |
| `poc_direction` | +0.53 | POC moving upward signals bullish momentum |
| `dist_prev_vah` | +0.51 | Opening near VAH increases acceptance probability |
| `dist_prev_val` | +0.23 | Opening far from VAL is mildly bullish |
| `prev_total_volume` | -0.83 | High volume days tend to be followed by consolidation |
| `dist_prev_poc` | -0.50 | Opening too far above POC suggests overextension |
| `prev_va_width` | -0.24 | Wide value areas reduce next day acceptance probability |

### Data Flow

<img width="718" height="209" alt="image" src="https://github.com/user-attachments/assets/11199bcc-a9c5-4847-9993-bb7fb88c631a" />


---

## evaluation.py

This module provides a deeper evaluation of the trained model beyond just accuracy, including a ROC curve and AUC score.

### Results

| Metric | Value |
|---|---|
| Accuracy | 86.3% |
| Baseline accuracy | 63.9% |
| Improvement over baseline | 22.4% |
| AUC Score | 0.947 |

An AUC of 0.947 is outstanding — 0.5 represents random guessing and 1.0 is a perfect model. Anything above 0.8 is considered strong for financial data.

The ROC curve shoots to nearly 1.0 true positive rate before the false positive rate even reaches 0.2, confirming the model has learned genuine structure from the VP features rather than just memorizing the training data.

### Trading Implication

- When the model predicts **no VAH acceptance** — 93% precision means this is a high confidence signal. Fading rallies toward the VAH on these days is well supported.
- When the model predicts **yes VAH acceptance** — 64% precision is decent but not as strong. Use as a directional bias combined with your own chart reading rather than acting on it blindly.

### Data Flow

<img width="417" height="295" alt="image" src="https://github.com/user-attachments/assets/88b9852b-f902-413c-8f7c-953d403b116b" />


---

## Future Improvements

The following enhancements are planned for future iterations of this project, ordered by priority. The backtesting dashboard comes first deliberately — before improving or expanding the model, it is essential to validate whether the current model is actually tradeable in practice. Statistical metrics like accuracy and AUC look great on paper but they don't tell you what would have happened to your account if you had followed every signal. The backtest answers that question first.

1. **Backtesting dashboard** — build an interactive dashboard showing the day by day performance of the model's signals over the 6 month study period. This includes cumulative returns, win rate per month, maximum drawdown, Sharpe ratio, and signals overlaid on the price chart. This validates whether the model is tradeable before any further development.

2. **Naked POC tracking** — identify unviolated previous POCs that price has never traded through since they were formed, and add their distance from current price as features. In VP theory these act as magnets and tend to get revisited.

3. **Additional labels** — expand beyond VAH acceptance to include POC acceptance, VAL rejection, and value area breakout/breakdown scenarios.

4. **More complex models** — test decision trees and gradient boosting against the logistic regression baseline to measure whether additional model complexity improves the edge.

5. **Tick data precision** — upgrade the VP computation to use raw ticks directly rather than $10 bucket aggregation for even more precise POC, VAH and VAL levels.

6. **Live signal generator** — connect the trained model to live Binance data to generate real time daily bias signals before the session opens.

7. **Automation** — connect to the Binance API to place orders programmatically if the model edge is proven robust enough over a longer period.

























































