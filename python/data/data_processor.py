import numpy as np
import pandas as pd
from pathlib import Path


class DataProcessor:
    """Load and process stock data for Markowitz portfolio optimization."""
    
    def __init__(self, data_path=None):
        """Initialize the data processor.
        
        Args:
            data_path: Path to the CSV file. If None, uses the default path.
        """
        if data_path is None:
            data_path = Path(__file__).parent / "all_stocks_5yr.csv"
        self.data_path = Path(data_path)
        self._raw_data = None
        self._prices = None
        self._returns = None
    
    def load_data(self):
        """Load the raw stock data from CSV."""
        self._raw_data = pd.read_csv(self.data_path)
        self._raw_data['date'] = pd.to_datetime(self._raw_data['date'])
        return self
    
    def get_stock_names(self):
        """Get list of all available stock names."""
        if self._raw_data is None:
            self.load_data()
        return sorted(self._raw_data['Name'].unique().tolist())
    
    def get_prices(self, stocks=None, start_date=None, end_date=None):
        """Get closing prices for selected stocks.
        
        Args:
            stocks: List of stock symbols. If None, uses all stocks.
            start_date: Start date (string or datetime). If None, uses first available.
            end_date: End date (string or datetime). If None, uses last available.
            
        Returns:
            DataFrame with dates as index and stocks as columns.
        """
        if self._raw_data is None:
            self.load_data()
        
        df = self._raw_data.copy()
        
        # Filter by date
        if start_date is not None:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # Filter by stocks
        if stocks is not None:
            df = df[df['Name'].isin(stocks)]
        
        # Pivot to get prices matrix (dates x stocks)
        prices = df.pivot(index='date', columns='Name', values='close')
        
        # Drop stocks with missing values
        prices = prices.dropna(axis=1)
        
        self._prices = prices
        return prices
    
    def compute_returns(self, prices=None):
        """Compute simple returns from prices.
        
        Returns: r_t = (p_t - p_{t-1}) / p_{t-1}
        
        Args:
            prices: DataFrame of prices. If None, uses stored prices.
            
        Returns:
            DataFrame of returns (one row less than prices).
        """
        if prices is None:
            if self._prices is None:
                raise ValueError("No prices available. Call get_prices() first.")
            prices = self._prices
        
        # Simple returns: (p_t - p_{t-1}) / p_{t-1}
        returns = prices.pct_change().dropna()
        
        self._returns = returns
        return returns
    
    def compute_mu_sigma(self, returns=None):
        """Compute expected returns (mu) and covariance matrix (Sigma).
        
        Args:
            returns: DataFrame of returns. If None, uses stored returns.
            
        Returns:
            tuple: (mu, sigma) where
                - mu: np.ndarray of shape (n,) - expected returns
                - sigma: np.ndarray of shape (n, n) - covariance matrix
        """
        if returns is None:
            if self._returns is None:
                raise ValueError("No returns available. Call compute_returns() first.")
            returns = self._returns
        
        # Expected returns (sample mean)
        mu = returns.mean().values
        
        # Covariance matrix (sample covariance)
        sigma = returns.cov().values
        
        return mu, sigma
    
    def get_optimization_data(self, stocks=None, start_date=None, end_date=None, n_stocks=None):
        """Convenience method to get mu and sigma for optimization.
        
        Args:
            stocks: List of stock symbols. If None, uses all stocks.
            start_date: Start date for data.
            end_date: End date for data.
            n_stocks: If specified, randomly select this many stocks.
            
        Returns:
            dict with keys: 'mu', 'sigma', 'stock_names', 'n_stocks', 'n_days'
        """
        if self._raw_data is None:
            self.load_data()
        
        # Handle stock selection
        if stocks is None and n_stocks is not None:
            all_stocks = self.get_stock_names()
            stocks = np.random.choice(all_stocks, size=min(n_stocks, len(all_stocks)), replace=False).tolist()
        
        # Get prices and compute returns
        prices = self.get_prices(stocks=stocks, start_date=start_date, end_date=end_date)
        returns = self.compute_returns(prices)
        mu, sigma = self.compute_mu_sigma(returns)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'stock_names': prices.columns.tolist(),
            'n_stocks': len(mu),
            'n_days': len(returns),
            'prices': prices,
            'returns': returns,
        }


def load_data(stocks=None, start_date=None, end_date=None, n_stocks=None, data_path=None):
    """Convenience function to load data and compute mu, sigma.
    
    Args:
        stocks: List of stock symbols. If None, uses all or n_stocks random ones.
        start_date: Start date for data.
        end_date: End date for data.
        n_stocks: If specified, randomly select this many stocks.
        data_path: Path to CSV file. If None, uses default.
        
    Returns:
        dict with keys: 'mu', 'sigma', 'stock_names', 'n_stocks', 'n_days', 'prices', 'returns'
    """
    processor = DataProcessor(data_path)
    return processor.get_optimization_data(
        stocks=stocks,
        start_date=start_date,
        end_date=end_date,
        n_stocks=n_stocks
    )


def get_initial_portfolio(n, method="uniform"):
    """Generate an initial portfolio on the simplex.
    
    Args:
        n: Number of assets.
        method: "uniform" for equal weights, "random" for random simplex point.
        
    Returns:
        np.ndarray of shape (n,) summing to 1.
    """
    if method == "uniform":
        return np.ones(n) / n
    elif method == "random":
        # Generate random point on simplex using Dirichlet distribution
        return np.random.dirichlet(np.ones(n))
    else:
        raise ValueError(f"Unknown method: {method}")