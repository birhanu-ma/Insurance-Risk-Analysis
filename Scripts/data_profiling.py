import pandas as pd

class DataProfiling:

    def __init__(self, filepath):
        """
        Initializes the DataProfiling object and loads the dataset.
        """
        self.filepath = filepath
        self.df = self.load_data()
    
    def load_data(self):
        """Loads a dataset into a pandas DataFrame."""
        try:
            df = pd.read_csv(self.filepath, delimiter="|", engine="python")
            print(f"📂 Dataset loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def check_dtypes(self):
        """Displays data types of each column."""
        print("\n🔹 Data Types:")
        print(self.df.dtypes)

    def check_missing_values(self):
        """Reports missing values for each column."""
        print("\n🔹 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values detected!")

    def check_duplicates(self):
        """Counts duplicate rows."""
        duplicates = self.df.duplicated().sum()
        print(f"\n🔹 Duplicate Rows: {duplicates}")
        if duplicates > 0:
            print("💡Consider removing duplicates using: df.drop_duplicates()")

    def basic_statistics(self):
        """Displays numerical summary statistics."""
        print("\n🔹 Summary Statistics:")
        print(self.df.describe())
