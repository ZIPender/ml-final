# YouTube Timestamp Generator

A machine learning-based tool that automatically generates timestamp chapters for YouTube videos.

## Features

- Extract transcripts from YouTube videos
- Generate timestamped chapters (coming soon)
- Clean, modern web interface
- FastAPI backend with automatic API documentation

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm

### Run the Application

1. **Clone or navigate to the project directory**
   ```bash
   cd youtube-timestamp
   ```

2. **Run the application with one command**
   ```bash
   ./run.sh
   ```

   This script will:
   - Set up Python virtual environment
   - Install all dependencies
   - Start both backend and frontend servers
   - Open the application in your browser

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

4. **Stop the application**
   - Press `Ctrl+C` in the terminal to stop both servers

## Manual Setup (Alternative)

If you prefer to run servers manually:

### Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start backend server
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd src

# Install dependencies
npm install

# Start frontend server
npm run dev
```

## Usage

1. Open http://localhost:3000 in your browser
2. Paste a YouTube video URL in the input field
3. Click "Get Transcript" to extract the video transcript
4. (Coming soon) View generated timestamp chapters

## Project Structure

```
youtube-timestamp/
├── backend/              # FastAPI backend
│   ├── app/
│   │   └── main.py      # API endpoints
│   └── requirements.txt  # Python dependencies
├── src/                  # Next.js frontend
│   ├── app/             # App Router pages
│   └── components/      # React components
├── run.sh               # Quick start script
└── README.md           # This file
```

## API Endpoints

- `GET /transcript?url={youtube_url}` - Extract transcript from YouTube video

## Development

The application uses:
- **Backend**: FastAPI, Python
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **ML**: Coming soon - Transformer models for chapter generation

## License

MIT 