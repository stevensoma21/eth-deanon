# Ethereum Transaction Analysis

A modular Python package for analyzing Ethereum transaction data to identify patterns and behavior, with a focus on deanonymization potential.

## Features

- Transaction data loading and preprocessing
- Network graph construction and analysis
- Entropy-based behavior analysis
- Static and temporal clustering
- Cohort analysis and evolution tracking
- Comprehensive visualization suite
- Statistical analysis of behavioral patterns

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eth-deanon.git
cd eth-deanon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials:
- Place your Google Cloud credentials JSON file in the appropriate location
- Update the path in `main.py` if necessary

## Usage

The main analysis pipeline can be run using:

```python
from eth_deanon.main import main

main()
```

## Project Structure

```
eth_deanon/
├── config/         # Configuration settings
├── data/          # Data loading and preprocessing
├── analysis/      # Analysis modules
├── visualization/ # Visualization functions
└── utils/         # Utility functions
```

## Modules

### Data Loading (`data/`)
- Transaction data loading from BigQuery
- Data preprocessing and normalization
- Address standardization

### Analysis (`analysis/`)
- Entropy calculation and analysis
- Clustering algorithms (DBSCAN, HDBSCAN)
- Temporal analysis
- Cohort analysis

### Visualization (`visualization/`)
- Cluster visualization
- Cohort comparison plots
- Temporal evolution plots
- Network graph visualization

## Configuration

Configuration parameters can be adjusted in `config/settings.py`:
- Data loading parameters
- Clustering parameters
- Visualization settings
- Analysis thresholds

## Output

The analysis generates:
- CSV files with analysis results
- PNG files with visualizations
- HTML files for interactive visualizations
- Comprehensive analysis report

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 