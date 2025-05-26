# Soil Classification Challenge - Annam.ai

## Team Information
- **Team Name:** SoilClassifiers
- **Organization:** Annam.ai IIT Ropar
- **Team Members:** 
  - Siddhant Bhardwaj
  - Sivadhanushya
- **Competition Rank:** 36

## Project Structure
```
.
├── .github/
│   └── workflows/
│       └── push.yml
├── challenge-1/
│   ├── data/
│   │   └── download.sh
│   ├── docs/
│   │   └── cards/
│   │       └── ml-metrics.json
│   ├── notebooks/
│   │   ├── training.ipynb
│   │   └── inference.ipynb
│   ├── requirements.txt
│   └── src/
│       ├── postprocessing.py
│       └── preprocessing.py
├── challenge-2/
│   ├── data/
│   │   └── download.sh
│   ├── docs/
│   │   └── cards/
│   │       └── ml-metrics.json
│   ├── notebooks/
│   │   ├── training.ipynb
│   │   └── inference.ipynb
│   ├── requirements.txt
│   ├── README.md
│   ├── postprocessing.py
│   └── src/
│       ├── postprocessing.py
│       └── preprocessing.py
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/CodeSasuke/soil-classification-challenge_annam
cd Soil-classification-challenege-annam
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r challenge-1/requirements.txt
```

## Model Performance
- F1 Score: 0.9009
- Platform: Kaggle (Username: annam.ai)

## Running the Code

### Challenge 1: Model Training
See detailed instructions in `challenge-1/README.md`

Required Dependencies:
- torch >= 1.10.0
- torchvision >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- And other packages listed in requirements.txt

### Challenge 2: Post-processing
See detailed instructions in `challenge-2/README.md`

Main features:
- Prediction formatting
- Submission file generation
- Label distribution analysis

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[Specify your license information here]
