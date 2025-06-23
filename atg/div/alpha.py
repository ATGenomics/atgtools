import numpy as np
import pandas as pd


class AlphaDiversity:
    """
    Calculates alpha diversity indices for a given species abundance profile.
    The input DataFrame should have species/OTUs as rows, and columns should include
    a species identifier, optionally grouping levels, and then sample abundance data.
    """

    @staticmethod
    def _get_valid_proportions(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get valid proportions from abundance array.
        
        Parameters
        ----------
        y : np.ndarray
            Array of species abundances
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (valid abundances, positive proportions)
        """
        valid_y = y[~np.isnan(y)]
        sum_valid_y = np.sum(valid_y)
        
        if sum_valid_y == 0:
            return valid_y, np.array([])
            
        proportions = valid_y / sum_valid_y
        positive_proportions = proportions[proportions > 0]
        return valid_y, positive_proportions

    @staticmethod
    def shannon(y: np.ndarray) -> float:
        """Calculates Shannon's H (natural logarithm, base e). H = -sum(p_i * log(p_i))."""
        _, positive_proportions = AlphaDiversity._get_valid_proportions(y)
        
        if positive_proportions.size == 0:
            return 0.0
            
        return -np.sum(positive_proportions * np.log(positive_proportions))

    @staticmethod
    def shannon2(y: np.ndarray, base: float = 2) -> float:
        """Calculates Shannon's H with a custom base. H_base = H_natural / log(base)."""
        if base <= 0 or base == 1:
            raise ValueError("Base for logarithm must be > 0 and not equal to 1.")
        h_natural = AlphaDiversity.shannon(y)
        return h_natural / np.log(base)

    @staticmethod
    def gini(y: np.ndarray) -> float:
        """Calculates Gini-Simpson index. D = 1 - sum(p_i^2)."""
        _, positive_proportions = AlphaDiversity._get_valid_proportions(y)
        
        if positive_proportions.size == 0:
            return 1.0

        return 1.0 - np.sum(positive_proportions**2)

    @staticmethod
    def simpson(y: np.ndarray) -> float:
        """Calculates Simpson's D (original). D = sum(p_i^2)."""
        _, positive_proportions = AlphaDiversity._get_valid_proportions(y)
        
        if positive_proportions.size == 0:
            return 0.0

        return np.sum(positive_proportions**2)

    @staticmethod
    def dom(y: np.ndarray) -> float:
        """Calculates Dominance index. D = max(p_i)."""
        _, positive_proportions = AlphaDiversity._get_valid_proportions(y)
        
        if positive_proportions.size == 0:
            return 0.0
            
        return np.max(positive_proportions)

    @staticmethod
    def richness(y: np.ndarray) -> float:
        """Calculates Species richness. S = Number of non-zero, non-NaN observations."""
        valid_y, _ = AlphaDiversity._get_valid_proportions(y)
        return float(np.sum(valid_y > 0))

    @staticmethod
    def pielou_e(y: np.ndarray) -> float:
        """Calculates Pielou's J (Evenness). J = H_natural / ln(S)."""
        valid_y, _ = AlphaDiversity._get_valid_proportions(y)
        
        if np.sum(valid_y) == 0:
            return 0.0
            
        S = AlphaDiversity.richness(y)
        if S <= 1:
            return float(S)
        
        H_natural = AlphaDiversity.shannon(y)
        return H_natural / np.log(S)

    def __init__(self, species_profile_df: pd.DataFrame, 
                 species_identifier_col: str = "taxonomy", 
                 group_by_level: str = None):
        """
        Initializes the AlphaDiversity calculator with species abundance data.

        Parameters
        ----------
        species_profile_df : pd.DataFrame
            DataFrame with species/OTUs as rows. It must contain:
            - A column for species identification (specified by `species_identifier_col`).
            - Optionally, a column for a higher taxonomic level to group by (specified by `group_by_level`).
            - One or more numeric columns representing sample abundances.
        species_identifier_col : str, optional
            Name of the column containing species/OTU identifiers, by default "taxonomy".
        group_by_level : str, optional
            Name of the column representing a taxonomic level to group species by before calculation,
            by default None (no grouping).
        """
        if not isinstance(species_profile_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if species_identifier_col not in species_profile_df.columns:
            raise ValueError("Species identifier column not found")

        metadata_cols = {species_identifier_col}
        if group_by_level:
            if group_by_level not in species_profile_df.columns:
                raise ValueError(f"Grouping level column '{group_by_level}' not found.")
            metadata_cols.add(group_by_level)
        
        sample_cols = []
        for col in species_profile_df.columns:
            if col not in metadata_cols:
                try:
                    pd.to_numeric(species_profile_df[col])
                    sample_cols.append(col)
                except (ValueError, TypeError):
                    continue
        if not sample_cols:
            raise ValueError("must be numeric")

        if group_by_level:
            grouped_data = species_profile_df.groupby(group_by_level)[sample_cols].sum()
            self.processed_df = grouped_data.T 
        else:
            if not species_profile_df[species_identifier_col].is_unique:
                species_data = species_profile_df.groupby(species_identifier_col)[sample_cols].sum()
            else:
                species_data = species_profile_df.set_index(species_identifier_col)[sample_cols]
            self.processed_df = species_data.T 

        self.sample_names = self.processed_df.index.tolist()
        self._validate_processed_data()
        self.normalized_abundance_matrix = self._normalize_data()

        self._shannon2_default_base = 2.0
        self._index_methods_map = {
            "Shannon": AlphaDiversity.shannon,
            "Shannon2": lambda y_arr: AlphaDiversity.shannon2(y_arr, base=self._shannon2_default_base),
            "Gini-Simpson": AlphaDiversity.gini,
            "Simpson": AlphaDiversity.simpson,
            "Dominance": AlphaDiversity.dom,
            "Richness": AlphaDiversity.richness,
            "Pielou": AlphaDiversity.pielou_e,
        }

    def _validate_processed_data(self) -> None:
        for col_name in self.processed_df.columns:
            if not pd.api.types.is_numeric_dtype(self.processed_df[col_name]):
                raise ValueError("Column must be numeric")
        if (self.processed_df < 0).any().any():
            raise ValueError("DataFrame contains negative values")

    def _normalize_data(self) -> np.ndarray:
        data_array = np.array(self.processed_df, dtype=float)
        row_sums = data_array.sum(axis=1)
        
        normalized_matrix = np.zeros_like(data_array)
        valid_rows_mask = row_sums > 0
        if np.any(valid_rows_mask):
            normalized_matrix[valid_rows_mask] = \
                data_array[valid_rows_mask] / row_sums[valid_rows_mask][:, np.newaxis]
        return normalized_matrix

    def calculate_diversity_indices(self, num_equiv: bool = True) -> pd.DataFrame:
        """
        Computes multiple alpha diversity indices for the processed data.

        Parameters
        ----------
        num_equiv : bool, optional
            If True, converts diversity indices to their number equivalents 
            (effective number of species). Default is True.
            - Shannon: np.exp(H)
            - Shannon2 (base B): B ** H_B
            - Gini-Simpson: 1 / (1 - D_gs)  (which is 1 / sum(p_i^2))
            - Simpson: 1 / D_s

        Returns
        -------
        pd.DataFrame
            A DataFrame with samples as rows and calculated diversity indices as columns.
        """
        all_diversity_results = []
        for index_name, div_func in self._index_methods_map.items():
            div_column_values = np.apply_along_axis(div_func, 1, self.normalized_abundance_matrix)

            if num_equiv:
                if index_name == "Shannon":
                    div_column_values = np.where(div_column_values == 0, 0, np.exp(div_column_values))
                elif index_name == "Shannon2":
                    div_column_values = self._shannon2_default_base ** div_column_values
                elif index_name == "Gini-Simpson":
                    # For Gini-Simpson, when D_gs = 1 (maximum diversity), number equivalent is infinity
                    # When D_gs = 0 (minimum diversity), number equivalent is 1
                    mask_max_div = np.isclose(div_column_values, 1.0)
                    mask_min_div = np.isclose(div_column_values, 0.0)
                    mask_normal = ~(mask_max_div | mask_min_div)
                    
                    result = np.zeros_like(div_column_values)
                    result[mask_max_div] = np.inf
                    result[mask_min_div] = 1.0
                    result[mask_normal] = 1.0 / (1.0 - div_column_values[mask_normal])
                    div_column_values = result
                    
                elif index_name == "Simpson":
                    # For Simpson, when D_s = 0 (maximum diversity), number equivalent is infinity
                    # When D_s = 1 (minimum diversity), number equivalent is 1
                    mask_max_div = np.isclose(div_column_values, 0.0)
                    mask_min_div = np.isclose(div_column_values, 1.0)
                    mask_normal = ~(mask_max_div | mask_min_div)
                    
                    result = np.zeros_like(div_column_values)
                    result[mask_max_div] = np.inf
                    result[mask_min_div] = 1.0
                    result[mask_normal] = 1.0 / div_column_values[mask_normal]
                    div_column_values = result
            
            all_diversity_results.append(
                pd.DataFrame(div_column_values, index=self.sample_names, columns=[index_name])
            )
        
        return pd.concat(all_diversity_results, axis=1)
