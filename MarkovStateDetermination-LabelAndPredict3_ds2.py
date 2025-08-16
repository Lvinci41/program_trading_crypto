import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from hmmlearn import hmm
import talib
import re

class MarketRegimeAnalyzer:
    def __init__(self, ohlc_df, lookback_window=20, volatility_threshold=0.5, chop_threshold=0.5):
        """
        Initialize the Market Regime Analyzer
        
        Parameters:
        - ohlc_df: DataFrame with Open, High, Low, Close, Volume columns (case/space insensitive)
        - lookback_window: Window for calculating rolling features
        - volatility_threshold: Threshold for identifying steady periods
        - chop_threshold: Threshold for identifying choppy periods
        """
        # Normalize column names first
        self.df = self.normalize_column_names(ohlc_df.copy())
        
        # Validate we have required columns
        self.validate_input_columns()
        
        self.lookback_window = lookback_window
        self.volatility_threshold = volatility_threshold
        self.chop_threshold = chop_threshold
        self.state_labels = {
            0: 'Rising',
            1: 'Falling',
            2: 'Steady',
            3: 'Choppy',
            4: 'No Label'
        }
        
    def normalize_column_names(self, df):
        """Normalize column names to lowercase, strip whitespace, and handle common variants"""
        # Create mapping of normalized name to original name
        column_mapping = {}
        for col in df.columns:
            normalized = col.strip().lower()
            # Handle common variants
            normalized = re.sub(r'[^a-z0-9]', '', normalized)
            column_mapping[col] = normalized
            
        # Identify required columns and their possible variants
        required_columns = {
            'open': ['open', 'op', 'o'],
            'high': ['high', 'hi', 'h'],
            'low': ['low', 'lo', 'l'],
            'close': ['close', 'cl', 'c', 'last'],
            'volume': ['volume', 'vol', 'v', 'qty']
        }
        
        # Find matches for each required column
        final_mapping = {}
        available_columns = set(column_mapping.values())
        
        for standard_name, variants in required_columns.items():
            for variant in variants:
                if variant in available_columns:
                    # Find original column name that matches this variant
                    for orig_col, normalized_col in column_mapping.items():
                        if normalized_col == variant:
                            final_mapping[standard_name] = orig_col
                            break
                    break
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in final_mapping.items()})
        
        return df
    
    def validate_input_columns(self):
        """Validate that we have all required columns after normalization"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self.df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns after normalization: {missing}")
        
        # Ensure numeric data types
        for col in required_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Drop rows with NaN in required columns
        self.df = self.df.dropna(subset=required_columns)
        
    def preprocess_data(self):
        """Calculate technical indicators and features"""
        # Calculate returns
        self.df['returns'] = self.df['close'].pct_change()
        
        # Calculate volatility measures
        self.df['volatility'] = self.df['close'].rolling(window=self.lookback_window).std()
        self.df['avg_volatility'] = self.df['volatility'].rolling(window=self.lookback_window).mean()
        self.df['norm_volatility'] = self.df['volatility'] / self.df['avg_volatility']
        
        # Calculate trend indicators
        self.df['sma'] = talib.SMA(self.df['close'], timeperiod=self.lookback_window)
        self.df['ema'] = talib.EMA(self.df['close'], timeperiod=self.lookback_window)
        self.df['adx'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.lookback_window)
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=self.lookback_window)
        
        # Calculate volume indicators
        self.df['volume_sma'] = talib.SMA(self.df['volume'], timeperiod=self.lookback_window)
        self.df['volume_change'] = self.df['volume'] / self.df['volume_sma']
        
        # Calculate price extremes
        self.df['local_min'] = self.df['close'] == self.df['close'].rolling(window=5, center=True).min()
        self.df['local_max'] = self.df['close'] == self.df['close'].rolling(window=5, center=True).max()
        
        # Calculate choppiness index
        self.df['atr'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.lookback_window)
        self.df['chop'] = 100 * np.log10(self.df['atr'].rolling(window=self.lookback_window).sum() / 
                                       (self.df['high'].rolling(window=self.lookback_window).max() - 
                                        self.df['low'].rolling(window=self.lookback_window).min())) / np.log10(self.lookback_window)
        
        # Remove remaining NaN values
        self.df = self.df.dropna()
        
    def label_states(self):
        """Label each observation with market regime states"""
        # Initialize state column
        self.df['state'] = 4  # Default to 'No Label'
        
        # Find all local minima and maxima
        min_indices = argrelextrema(self.df['close'].values, np.less_equal, order=5)[0]
        max_indices = argrelextrema(self.df['close'].values, np.greater_equal, order=5)[0]
        
        # Label rising periods (from local min to local max)
        for i in range(len(min_indices)-1):
            start_idx = min_indices[i]
            end_idx = max_indices[max_indices > start_idx][0] if any(max_indices > start_idx) else len(self.df)-1
            
            # Check if price stays above the initial low
            if all(self.df['close'].iloc[start_idx:end_idx+1] >= self.df['close'].iloc[start_idx]):
                self.df['state'].iloc[start_idx:end_idx+1] = 0  # Rising
        
        # Label falling periods (from local max to local min)
        for i in range(len(max_indices)-1):
            start_idx = max_indices[i]
            end_idx = min_indices[min_indices > start_idx][0] if any(min_indices > start_idx) else len(self.df)-1
            
            # Check if price stays below the initial high
            if all(self.df['close'].iloc[start_idx:end_idx+1] <= self.df['close'].iloc[start_idx]):
                self.df['state'].iloc[start_idx:end_idx+1] = 1  # Falling
        
        # Label steady periods (low volatility)
        steady_mask = (self.df['norm_volatility'] < self.volatility_threshold) & \
                     (self.df['state'] == 4)  # Only override unlabeled periods
        self.df.loc[steady_mask, 'state'] = 2  # Steady
        
        # Label choppy periods (high chop index but not trending)
        chop_mask = (self.df['chop'] > self.chop_threshold) & \
                    (self.df['adx'] < 25) & \
                    (self.df['state'] == 4)  # Only override unlabeled periods
        self.df.loc[chop_mask, 'state'] = 3  # Choppy
        
    def prepare_model_data(self):
        """Prepare features and labels for machine learning models"""
        # Create lagged features
        feature_cols = ['returns', 'volatility', 'norm_volatility', 'sma', 'ema',
                        'adx', 'rsi', 'volume_change', 'chop']
        for col in feature_cols:
            for lag in range(1, 4):  # Create 3 lags for each feature
                self.df[f'{col}_lag{lag}'] = self.df[col].shift(lag)
        # Create moving averages of features
        for col in feature_cols:
            self.df[f'{col}_ma5'] = self.df[col].rolling(5).mean()
            self.df[f'{col}_ma10'] = self.df[col].rolling(10).mean()
        # Drop rows with NaN values (from lagged features)
        self.df = self.df.dropna()

        # EXCLUDE NON-NUMERIC COLUMNS (the key fix!)
        # Only keep numeric columns, drop 'state', 'local_min', 'local_max', and any datetime/object columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['state', 'local_min', 'local_max']
        model_features = [col for col in numeric_cols if col not in exclude_cols]
        self.X = self.df[model_features]
        self.y = self.df['state']

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False)

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_models(self):
        """Train logistic regression, random forest, and neural network models"""
        # Logistic Regression
        self.lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        lr_pred = self.lr_model.predict(self.X_test_scaled)
        print("Logistic Regression Performance:")
        print(classification_report(self.y_test, lr_pred, target_names=self.state_labels.values()))
        
        # Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        rf_pred = self.rf_model.predict(self.X_test)
        print("\nRandom Forest Performance:")
        print(classification_report(self.y_test, rf_pred, target_names=self.state_labels.values()))
        
        # Neural Network
        self.nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', 
                                     solver='adam', max_iter=1000, random_state=42)
        self.nn_model.fit(self.X_train_scaled, self.y_train)
        nn_pred = self.nn_model.predict(self.X_test_scaled)
        print("\nNeural Network Performance:")
        print(classification_report(self.y_test, nn_pred, target_names=self.state_labels.values()))
    
    def analyze_state_transitions(self):
        """Analyze transition probabilities between states"""

        # Fit Hidden Markov Model
        self.hmm_model = hmm.CategoricalHMM(n_components=5, n_iter=100)
        self.hmm_model.fit(self.y.values.reshape(-1, 1))

        # Get transition matrix
        transition_matrix = self.hmm_model.transmat_
        print("\nState Transition Probabilities:")
        transition_df = pd.DataFrame(
            transition_matrix,
            index=self.state_labels.values(),
            columns=self.state_labels.values()
        )
        print(transition_df.round(3))

        # Calculate empirical transition probabilities
        # Corrected block for transitions
        prev_state = self.df['state'].shift()
        curr_state = self.df['state']
        mask = (prev_state != curr_state) & prev_state.notna()
        transitions = pd.crosstab(prev_state[mask], curr_state[mask])
        empirical_transitions = transitions.div(transitions.sum(axis=1), axis=0)
        print("\nEmpirical Transition Probabilities:")
        print(empirical_transitions.round(3))

    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting market regime analysis...")
        self.preprocess_data()
        print("Data preprocessing completed.")
        
        self.label_states()
        print(f"State labeling completed. Distribution:\n{self.df['state'].value_counts()}")
        
        self.prepare_model_data()
        print("Model data preparation completed.")
        
        self.train_models()
        print("Model training completed.")
        
        self.analyze_state_transitions()
        print("State transition analysis completed.")


# Example usage with messy column names
if __name__ == "__main__":
    # Create sample data with messy column names
    ohlc_data = pd.read_csv('20250202-20170908_BTC-USDT_1D.csv', skip_blank_lines=True, parse_dates=["timestamp"], dayfirst=False)
    
    print("Original columns:", ohlc_data.columns.tolist())
    
    # Run analysis
    analyzer = MarketRegimeAnalyzer(ohlc_data)
    analyzer.run_analysis()
    
    print("\nNormalized columns in processed data:", analyzer.df.columns.tolist())

