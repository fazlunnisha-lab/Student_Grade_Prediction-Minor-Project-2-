# Student_Grade_Prediction-Minor-Project-2-
ML-powered grade predictor using assignment scores &amp; attendance. Built with Python, Pandas, scikit-learn &amp; Streamlit. Identifies at-risk students early through interactive dashboards.
# GradeWise — Student Grade Predictor

*Because every student deserves to know where they stand before the final exam.*

## What This Is

GradeWise is a lightweight machine learning tool that predicts a student's final grade based on their assignment scores and attendance record. Built during my second year at IIT Madras, this project explores how simple data patterns can help educators identify at-risk students early and give them a chance to course-correct.

I built this because I have seen classmates panic two weeks before finals, wondering if they can still pass. This tool removes that guesswork. Input your scores. Know your standing. Act early.

## The Problem

Academic performance is rarely a surprise on result day—it is the sum of small signals scattered across the semester. Yet most students only connect the dots when it is too late. Teachers, too, lack quick tools to flag struggling students without manually crunching numbers.

GradeWise bridges that gap. It treats assignments and attendance as leading indicators, not just records, and turns them into actionable predictions.

## How It Works

1. **Upload or enter** your assignment scores (3 assignments) and attendance percentage
2. **The model** (trained on historical student data) predicts your likely final grade
3. **See your risk zone** — green (safe), yellow (borderline), or red (at-risk)
4. **Get suggestions** on which assignments to prioritize if you want to improve

## Features

- CSV upload or manual input for single-student prediction
- Real-time grade prediction with confidence interval
- Risk classification with color-coded feedback
- Batch analysis for entire classrooms
- Interactive visualizations: predicted vs. actual grade scatter, feature importance chart
- Sample dataset included for instant exploration

## Tech Stack

| Layer | Tool | Why I Chose It |
|---|---|---|
| Language | Python | Clean syntax, vast ML ecosystem |
| Data Processing | Pandas, NumPy | Fast manipulation of structured academic data |
| Machine Learning | scikit-learn | Production-ready regression with minimal overhead |
| Visualization | Plotly | Interactive charts that feel modern and responsive |
| Dashboard | Streamlit | From script to live app in under 30 minutes |
| Deployment | Streamlit Cloud | Free, automatic, GitHub-native |

## Project Structure
