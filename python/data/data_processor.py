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
            'stock_names': prices.columns.to_numpy(),
            'n_stocks': len(mu),
            'n_days': len(returns),
            'prices': prices,
            'returns': returns,
        }

    def get_train_test_split(self, train_end_date, stocks=None, n_stocks=None, 
                              train_start_date=None, test_end_date=None):
        """Split data into training and test sets by date.
        
        Args:
            train_end_date: End date for training data (exclusive for test).
            stocks: List of stock symbols. If None, uses all stocks.
            n_stocks: If specified, randomly select this many stocks.
            train_start_date: Start date for training data. If None, uses first available.
            test_end_date: End date for test data. If None, uses last available.
            
        Returns:
            tuple: (train_data, test_data) where each is a dict with keys:
                'mu', 'sigma', 'stock_names', 'n_stocks', 'n_days', 'prices', 'returns'
        """
        if self._raw_data is None:
            self.load_data()
        
        # Handle stock selection (same stocks for both train and test)
        if stocks is None and n_stocks is not None:
            all_stocks = self.get_stock_names()
            stocks = np.random.choice(all_stocks, size=min(n_stocks, len(all_stocks)), replace=False).tolist()
        
        # Get all prices first to ensure same stocks in both sets
        all_prices = self.get_prices(stocks=stocks, start_date=train_start_date, end_date=test_end_date)
        stock_list = all_prices.columns.tolist()
        
        # Split by date
        train_end = pd.to_datetime(train_end_date)
        
        train_prices = all_prices[all_prices.index < train_end]
        test_prices = all_prices[all_prices.index >= train_end]
        
        # Compute returns for each set
        train_returns = train_prices.pct_change().dropna()
        test_returns = test_prices.pct_change().dropna()
        
        # Compute mu and sigma from training data only
        train_mu = train_returns.mean().values
        train_sigma = train_returns.cov().values
        
        # Test set uses its own realized returns (for evaluation)
        test_mu = test_returns.mean().values
        test_sigma = test_returns.cov().values
        
        train_data = {
            'mu': train_mu,
            'sigma': train_sigma,
            'stock_names': np.array(stock_list),
            'n_stocks': len(stock_list),
            'n_days': len(train_returns),
            'prices': train_prices,
            'returns': train_returns,
        }
        
        test_data = {
            'mu': test_mu,  # Realized (out-of-sample) expected returns
            'sigma': test_sigma,  # Realized (out-of-sample) covariance
            'stock_names': np.array(stock_list),
            'n_stocks': len(stock_list),
            'n_days': len(test_returns),
            'prices': test_prices,
            'returns': test_returns,
        }
        
        return train_data, test_data


def evaluate_portfolio(w, returns):
    """Evaluate portfolio performance on given returns.
    
    Args:
        w: Portfolio weights (np.ndarray of shape (n,))
        returns: DataFrame of returns (T x n) or dict with 'returns' key
        
    Returns:
        dict with performance metrics
    """
    if isinstance(returns, dict):
        returns = returns['returns']
    
    # Portfolio returns over time: r_p(t) = w^T @ r(t)
    portfolio_returns = returns.values @ w
    
    # Basic statistics
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    total_return = np.prod(1 + portfolio_returns) - 1
    
    # Annualized metrics (assuming ~252 trading days)
    n_days = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_std = std_return * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    annualized_sharpe = annualized_return / annualized_std if annualized_std > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = np.max(drawdown)
    
    # Realized variance vs predicted (if sigma available)
    realized_variance = np.var(portfolio_returns)
    
    return {
        'mean_daily_return': mean_return,
        'std_daily_return': std_return,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_std': annualized_std,
        'sharpe_ratio': sharpe_ratio,
        'annualized_sharpe': annualized_sharpe,
        'max_drawdown': max_drawdown,
        'realized_variance': realized_variance,
        'n_days': n_days,
        'portfolio_returns': portfolio_returns,
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