#!/usr/bin/env python3
"""
PISCO Profile Processing Script

This script processes PISCO profiles for segmentation, classification, and EcoTaxa export.
It supports two modes:
1. Benchmark mode: Process profiles from multiple cruises (as per segment_benchmark_v3.py)
2. Cruise mode: Process all profiles from a specific cruise

Features:
- Toggle EcoTaxa zip export (--export-zip / --no-export-zip)
- Always saves df_ET as TSV/CSV regardless of zip export setting
- Command-line interface for easy configuration
"""

import os
import sys
import argparse
import socket
import datetime
import shutil
import logging
import re
from glob import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import traceback

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Image as ImageFeature, Value

from segmenter import run_segmenter
import analyze_profiles_seavision as ap


@dataclass
class ProfileInfo:
    """Normalized profile information across all cruise naming conventions."""
    profile_name: str
    profile_id: str
    cruise: str
    latitude: float
    longitude: float
    date_str: str  # YYYYMMDD
    time_str: str  # HHMMSS
    datetime_str: str  # YYYYMMDD-HHMMSS
    pressure_unit: str
    yaml_path: Optional[str] = None
    ctd_id: Optional[str] = None
    has_yaml: bool = False


class CruiseAdapter(ABC):
    """Base adapter for cruise-specific naming conventions."""
    
    def __init__(self, cruise_name: str, pressure_unit: str = "dbar"):
        self.cruise_name = cruise_name
        self.pressure_unit = pressure_unit
    
    @abstractmethod
    def parse_profile(self, profile_name: str, profile_path: str) -> ProfileInfo:
        """Parse cruise-specific profile information."""
        pass
    
    def extract_ctd_id(self, profile_name: str) -> Optional[str]:
        """Extract CTD profile ID (override if cruise-specific)."""
        return None


class BenchmarkV3Adapter(CruiseAdapter):
    """Adapter for benchmark_v3 dataset - inherits full cruise info from source."""
    
    def parse_profile(self, profile_name: str, profile_path: str) -> ProfileInfo:
        """Parse profile from benchmark_v3 (inherits original cruise structure)."""
        # Try to extract lat/lon from profile name
        lat, lon = self._extract_lat_lon_from_name(profile_name)
        
        # Try to find yaml file
        yaml_path = os.path.join(profile_path, profile_name + ".yaml")
        has_yaml = os.path.exists(yaml_path)
        
        if has_yaml:
            try:
                lat, lon = ap.extract_coords_from_yaml(yaml_path)
            except:
                pass
        
        # Extract datetime from profile name (assuming end of name: YYYYMMDD-HHMM)
        try:
            timestamp_part = profile_name.split('_')[-1]
            date_str = timestamp_part.split('-')[0]
            time_str = timestamp_part.split('-')[1] + '00'  # Add seconds
            datetime_str = f"{date_str}-{time_str}"
        except:
            # Fallback to placeholder
            date_str = "20260101"
            time_str = "120000"
            datetime_str = "20260101-120000"
        
        return ProfileInfo(
            profile_name=profile_name,
            profile_id=profile_name,
            cruise=self.cruise_name,
            latitude=lat,
            longitude=lon,
            date_str=date_str,
            time_str=time_str,
            datetime_str=datetime_str,
            pressure_unit=self.pressure_unit,
            yaml_path=yaml_path if has_yaml else None,
            has_yaml=has_yaml
        )
    
    def _extract_lat_lon_from_name(self, profile_name: str) -> Tuple[float, float]:
        """Extract coordinates from profile name."""
        try:
            return ap.extract_lat_lon_from_profile(profile_name)
        except:
            return 0.0, 0.0
    
    def extract_ctd_id(self, profile_name: str) -> Optional[str]:
        """Extract CTD ID from profile name."""
        # Try different cruise formats
        try:
            # Check if it's HE570 style
            if 'HE570' in profile_name:
                return profile_name.split('_')[1] + profile_name.split('_')[2]
            # Otherwise generic extraction
            return ap.get_ctd_profile_id(os.path.dirname(profile_name), profile_name)
        except:
            return None


class HE570Adapter(CruiseAdapter):
    """Adapter for HE570 cruise."""
    
    def parse_profile(self, profile_name: str, profile_path: str) -> ProfileInfo:
        yaml_path = os.path.join(profile_path, profile_name + ".yaml")
        has_yaml = os.path.exists(yaml_path)
        
        if has_yaml:
            lat, lon = ap.extract_coords_from_yaml(yaml_path)
        else:
            lat, lon = 0.0, 0.0
        
        # Split date-time
        try:
            datetime_str = profile_name.split('_')[-1]
            date_str, time_str = datetime_str.split('-')
        except:
            date_str = "20200101"
            time_str = "120000"
            datetime_str = "20200101-120000"
        
        # CTD ID
        try:
            ctd_id = profile_name.split('_')[1] + profile_name.split('_')[2]
        except:
            ctd_id = None
        
        return ProfileInfo(
            profile_name=profile_name,
            profile_id=ctd_id if ctd_id else profile_name,
            cruise=self.cruise_name,
            latitude=lat,
            longitude=lon,
            date_str=date_str,
            time_str=time_str,
            datetime_str=datetime_str,
            pressure_unit=self.pressure_unit,
            yaml_path=yaml_path if has_yaml else None,
            ctd_id=ctd_id,
            has_yaml=has_yaml
        )
    
    def extract_ctd_id(self, profile_name: str) -> Optional[str]:
        try:
            return profile_name.split('_')[1] + profile_name.split('_')[2]
        except:
            return None


class GenericCruiseAdapter(CruiseAdapter):
    """Generic adapter for most cruises (SO298, MSM126, M202, SO308)."""
    
    def parse_profile(self, profile_name: str, profile_path: str) -> ProfileInfo:
        # Check for yaml in Metadata subfolder
        yaml_path = os.path.join(profile_path, profile_name + "_Metadata", profile_name + ".yaml")
        has_yaml = os.path.exists(yaml_path)
        
        if has_yaml:
            lat, lon = ap.extract_coords_from_yaml(yaml_path)
        else:
            lat, lon = ap.extract_lat_lon_from_profile(profile_name)
        
        # Split date-time
        try:
            datetime_str = profile_name.split('_')[-1]
            date_str, time_str = datetime_str.split('-')
        except:
            date_str = "20200101"
            time_str = "120000"
            datetime_str = "20200101-120000"
        
        return ProfileInfo(
            profile_name=profile_name,
            profile_id=profile_name.split('_')[1] if '_' in profile_name else profile_name,
            cruise=self.cruise_name,
            latitude=lat,
            longitude=lon,
            date_str=date_str,
            time_str=time_str,
            datetime_str=datetime_str,
            pressure_unit=self.pressure_unit,
            yaml_path=yaml_path if has_yaml else None,
            has_yaml=has_yaml
        )
    
    def extract_ctd_id(self, profile_name: str) -> Optional[str]:
        try:
            return profile_name.split('_')[1]
        except:
            return None


class M181Adapter(CruiseAdapter):
    """Adapter for M181 cruise (uses bar instead of dbar)."""
    
    def __init__(self):
        super().__init__("M181", pressure_unit="bar")
    
    def parse_profile(self, profile_name: str, profile_path: str) -> ProfileInfo:
        lat, lon = ap.extract_lat_lon_from_profile(profile_name)
        
        # Split datetime
        try:
            datetime_str = profile_name.split('_')[0]
            date_str = datetime_str.split('-')[0]
            time_str = datetime_str.split('-')[1]
        except:
            date_str = "20220101"
            time_str = "120000"
            datetime_str = "20220101-120000"
        
        # CTD ID
        try:
            ctd_id = profile_name.split('_')[0].split('-')[1] + '-' + profile_name.split('_')[0].split('-')[2]
        except:
            ctd_id = None
        
        return ProfileInfo(
            profile_name=profile_name,
            profile_id=ctd_id if ctd_id else profile_name,
            cruise=self.cruise_name,
            latitude=lat,
            longitude=lon,
            date_str=date_str,
            time_str=time_str,
            datetime_str=datetime_str,
            pressure_unit=self.pressure_unit,
            yaml_path=None,
            ctd_id=ctd_id,
            has_yaml=False
        )
    
    def extract_ctd_id(self, profile_name: str) -> Optional[str]:
        try:
            return profile_name.split('_')[1].split('-')[1]
        except:
            return None


# Adapter registry
CRUISE_ADAPTERS: Dict[str, CruiseAdapter] = {
    "HE570": HE570Adapter("HE570"),
    "M181": M181Adapter(),
    "SO298": GenericCruiseAdapter("SO298"),
    "MSM126": GenericCruiseAdapter("MSM126"),
    "M202": GenericCruiseAdapter("M202"),
    "SO308": GenericCruiseAdapter("SO308"),
    "benchmark_v3": BenchmarkV3Adapter("benchmark_v3"),
}


def get_adapter_for_cruise(cruise: str) -> CruiseAdapter:
    """Get the appropriate adapter for a cruise."""
    return CRUISE_ADAPTERS.get(cruise, GenericCruiseAdapter(cruise))


class Logger:
    """Simple logger that writes to both terminal and file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        """Print to terminal and write to log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        self.log_file.close()


def find_image_dirs(profile_path: str, image_ext: str) -> List[str]:
    """Return directories in a profile that directly contain images."""
    candidate_dirs = []
    for item in os.listdir(profile_path):
        item_path = os.path.join(profile_path, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()
            if 'image' in item_lower or item_lower == 'png':
                candidate_dirs.append(item_path)

    candidate_dirs.append(profile_path)

    image_dirs = []
    for img_dir in candidate_dirs:
        if glob(os.path.join(img_dir, f"*{image_ext}")):
            image_dirs.append(img_dir)

    seen = set()
    deduped = []
    for img_dir in image_dirs:
        if img_dir not in seen:
            seen.add(img_dir)
            deduped.append(img_dir)

    return deduped


def select_equally_spaced(items: List[str], count: int) -> List[str]:
    """Select roughly equally spaced items, preserving order."""
    if count <= 0:
        return []
    if count >= len(items):
        return items
    if count == 1:
        return [items[0]]

    step = (len(items) - 1) / (count - 1)
    indices = [int(round(i * step)) for i in range(count)]

    selected = []
    seen = set()
    for idx in indices:
        idx = max(0, min(idx, len(items) - 1))
        if idx not in seen:
            seen.add(idx)
            selected.append(items[idx])

    if len(selected) < count:
        for i, item in enumerate(items):
            if i not in seen:
                selected.append(item)
                seen.add(i)
                if len(selected) == count:
                    break

    return selected


def process_profile_postanalysis(
    profile_name: str,
    results_folder: str,
    cruise: str,
    profile_path: str,
    logger: Logger,
    predict_ViT: bool = True,
    export_zip: bool = True,
    ctd_dir: Optional[str] = None,
    ctd_prefix: Optional[str] = None,
    log_directory: Optional[str] = None
) -> bool:
    """
    Process a profile for analysis, classification, and optional EcoTaxa export.
    Creates two separate EcoTaxa exports: one for non-living and one for all other classes.
    Uses cruise adapters to normalize naming conventions.
    
    Args:
        export_zip: If True, create EcoTaxa zip files. If False, only save df_ET as TSV/CSV.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.log(f"\n--- Post-analysis for {profile_name} ---")
        
        # Get the appropriate adapter for this cruise
        adapter = get_adapter_for_cruise(cruise)
        profile_info = adapter.parse_profile(profile_name, profile_path)
        
        logger.log(f"  Cruise: {profile_info.cruise}")
        logger.log(f"  Profile ID: {profile_info.profile_id}")
        logger.log(f"  Coordinates: {profile_info.latitude:.4f}, {profile_info.longitude:.4f}")
        logger.log(f"  Pressure unit: {profile_info.pressure_unit}")
        logger.log(f"  Export ZIP: {export_zip}")
        
        # Define paths
        profile_data_dir = os.path.join(results_folder, 'Data')
        ecotaxa_dir = os.path.join(results_folder, 'EcoTaxa')
        
        # Check if already processed (based on ecotaxa_dir existence)
        if os.path.exists(ecotaxa_dir):
            logger.log(f"EcoTaxa directory already exists, skipping.")
            return True
        
        # Generate crop dataframe
        logger.log(f'Generating crop dataframe...')
        if not os.path.exists(profile_data_dir):
            logger.log(f"Data directory {profile_data_dir} does not exist, skipping.")
            return False
        
        df = ap.gen_crop_df(
            profile_data_dir,
            small=False,
            size_filter=0,
            cruise=cruise,
            pressure_unit=profile_info.pressure_unit
        )
        logger.log(f"  {len(df.index)} particles found")
        
        # Add full frame paths
        df['fullframe_path'] = df['full_path'].apply(ap.modify_full_path)
        
        # Filter by aspect ratio
        threshold_ratio = 0.005
        removed_ar = len(df) - len(df[df['w'] >= threshold_ratio * df['h']])
        logger.log(f"  {removed_ar} crops removed due to aspect ratio")
        df = df[df['w'] >= threshold_ratio * df['h']]
        
        # Add CTD data if available
        if ctd_dir is not None and os.path.exists(ctd_dir):
            try:
                ctd_id = profile_info.ctd_id or adapter.extract_ctd_id(profile_name)
                
                if ctd_id:
                    ctd_file = os.path.join(ctd_dir, f'{ctd_prefix}{ctd_id}.ctd')
                    if os.path.exists(ctd_file):
                        logger.log(f'  Adding CTD data from {ctd_file}...')
                        df = ap.add_ctd_data(ctd_file, df)
                    else:
                        logger.log(f"  CTD file {ctd_file} not found")
            except Exception as e:
                logger.log(f"  CTD data addition failed: {e}")
        
        # Add log data if available
        if log_directory is not None and os.path.exists(log_directory):
            try:
                logger.log('  Adding log info...')
                # Use normalized datetime
                date_time_obj = datetime.datetime.strptime(
                    profile_info.datetime_str[:13], '%Y%m%d-%H%M'
                )
                min_diff = datetime.timedelta(days=365*1000)
                closest_file = None
                
                for filename in os.listdir(log_directory):
                    if '__Templog.txt' in filename:
                        file_timestamp = filename[:16]
                        file_datetime = datetime.datetime.strptime(
                            file_timestamp, '%Y%m%d_%Hh_%Mm'
                        )
                        diff = abs(date_time_obj - file_datetime)
                        if diff < min_diff:
                            min_diff = diff
                            closest_file = filename
                
                if closest_file:
                    file_path = os.path.join(log_directory, closest_file)
                    df_log = ap.create_log_df(file_path, cruise=cruise)
                    
                    # Merge log data
                    df.drop(['TT_x', 'T1_x', 'T2_x', 'TH_x', 'restart_x', 'relock_x', 'Time_log_x',
                             'TT_y', 'T1_y', 'T2_y', 'TH_y', 'restart_y', 'relock_y', 'Time_log_y',
                             'TT', 'T1', 'T2', 'TH', 'restart', 'relock', 'Time_log'],
                            axis=1, inplace=True, errors='ignore')
                    
                    df['timestamp'] = pd.to_datetime(df['date-time'], format='%Y%m%d-%H%M%S%f')
                    df = df.sort_values('timestamp')
                    df_log = df_log.sort_values('timestamp')
                    
                    df = pd.merge_asof(df, df_log, left_on='timestamp', right_on='timestamp', direction='backward')
                    df.drop('timestamp', axis=1, inplace=True)
                    
                    # Add particle count filter
                    required_columns = ['date-time', 'pressure [dbar]', 'img_id']
                    optional_columns = ['depth [m]', 'temperature', 'interpolated_s', 'interpolated_t',
                                      'interpolated_o', 'interpolated_z_factor', 'restart', 'relock', 'TAG_event']
                    existing_columns = [col for col in required_columns + optional_columns if col in df.columns]
                    
                    df_unique = df[existing_columns].drop_duplicates()
                    df_count = df.groupby('date-time').size().reset_index(name='count')
                    df_unique = df_unique.merge(df_count, on='date-time', how='left')
                    df_unique = df_unique.sort_values('pressure [dbar]')
                    df_unique['part_based_filter'] = df_unique['count'].apply(
                        lambda x: 0 if x < (df_unique['count'].mean() + 5*df_unique['count'].std()) else 1
                    )
                    df = df.merge(df_unique[['date-time', 'part_based_filter']], on='date-time', how='left')
                else:
                    logger.log("  Logfile not found")
            except Exception as e:
                logger.log(f"  Log data addition failed: {e}")
        
        logger.log(f"  {len(df.index)} particles after adding metadata")
        
        # Add object IDs and coordinates using normalized data
        df['object_id'] = profile_info.profile_id + df['img_id'].astype(str) + '_' + df['index'].astype(str)
        df['lat'] = profile_info.latitude
        df['lon'] = profile_info.longitude
        df['date'] = profile_info.date_str
        df['time'] = profile_info.time_str
        
        # Filter out anomalies
        if "TAG_event" in df.columns:
            removed_tag = len(df) - len(df[df['TAG_event'] == 0])
            logger.log(f"  {removed_tag} crops removed due to TAG events")
            df = df[df['TAG_event'] == 0]
        
        if "part_based_filter" in df.columns:
            removed_pbf = len(df) - len(df[df['part_based_filter'] == 0])
            logger.log(f"  {removed_pbf} crops removed due to particle count filter")
            df = df[df['part_based_filter'] == 0]
        
        # Remove duplicates
        df = df.drop_duplicates(subset='full_path', keep='first')
        df = df.sort_values(by='object_id')
        df.reset_index(drop=True, inplace=True)
        logger.log(f"  {len(df)} crops remaining after filtering")
        
        # ViT predictions
        if predict_ViT:
            logger.log(f"  Running ViT predictions...")
            prediction_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            prediction_host = socket.gethostname()
            prediction_env = sys.executable
            
            img_dir = os.path.join(results_folder, "Deconv_crops")
            if not os.path.exists(img_dir):
                logger.log(f"  Deconv_crops directory not found, skipping ViT predictions")
                predict_ViT = False
            else:
                try:
                    valid_filenames = set(df['filename'])
                    ds_pisco = ap.load_unclassified_images(img_dir, filenames=valid_filenames)
                    ds_pisco_trans = ds_pisco.with_transform(ap.process_batch)
                    
                    # Binary classifier
                    model_dir_bin = '/home/veit/PIScO_dev/ViT_custom_size_sensitive_binary/best_model'
                    filenames, predictions, probabilities, entropy_scores, ood_flags = \
                        ap.get_predictions_with_entropy_ood_binary(
                            ds_pisco_trans, model_dir_bin, entropy_threshold=0.5,
                            temperature=1.5, batch_size=64
                        )
                    
                    prediction_df = pd.DataFrame({
                        'filename': filenames,
                        'is_ood': ood_flags,
                        'entropy': entropy_scores,
                        'top1': [pred[0] for pred in predictions],
                        'prob1': [prob[0] for prob in probabilities],
                    })
                    
                    # Living classifier
                    living_filenames = prediction_df[prediction_df['top1'] == 'living']['filename'].tolist()
                    logger.log(f"  {len(living_filenames)} living particles predicted")
                    
                    if living_filenames:
                        living_data = {
                            'image': living_filenames,
                            'label': living_filenames
                        }
                        features = Features({
                            'image': ImageFeature(),
                            'label': Value('string')
                        })
                        ds_living = Dataset.from_dict(living_data, features=features)
                        ds_living_trans = ds_living.with_transform(ap.process_batch)
                        
                        model_dir_mc = '/home/veit/PIScO_dev/ViT_custom_size_sensitive_v5/best_model'
                        filenames_living, predictions_living, probabilities_living, \
                            entropy_scores_living, ood_flags_living = \
                            ap.get_predictions_with_entropy_ood(
                                ds_living_trans, model_dir_mc, entropy_threshold=0.9,
                                temperature=1.5, batch_size=64
                            )
                        
                        df_living = pd.DataFrame({
                            'filename': filenames_living,
                            'is_ood': ood_flags_living,
                            'entropy': entropy_scores_living,
                            'top1': [pred[0] for pred in predictions_living],
                            'top2': [pred[1] for pred in predictions_living],
                            'top3': [pred[2] for pred in predictions_living],
                            'top4': [pred[3] for pred in predictions_living],
                            'top5': [pred[4] for pred in predictions_living],
                            'prob1': [prob[0] for prob in probabilities_living],
                            'prob2': [prob[1] for prob in probabilities_living],
                            'prob3': [prob[2] for prob in probabilities_living],
                            'prob4': [prob[3] for prob in probabilities_living],
                            'prob5': [prob[4] for prob in probabilities_living]
                        })
                        
                        # Merge predictions
                        prediction_df_combined = prediction_df.copy()
                        df_living = df_living.set_index('filename')
                        prediction_df_combined = prediction_df_combined.set_index('filename')
                        for col in ['top1', 'top2', 'top3', 'top4', 'top5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'entropy', 'is_ood']:
                            prediction_df_combined.loc[df_living.index, col] = df_living[col]
                        prediction_df_combined = prediction_df_combined.reset_index()
                    else:
                        prediction_df_combined = prediction_df.copy()
                    
                    # Add metadata
                    prediction_df_combined['prediction_model'] = np.where(
                        prediction_df_combined['top1'] == 'living',
                        model_dir_mc if living_filenames else model_dir_bin,
                        model_dir_bin
                    )
                    prediction_df_combined['prediction_timestamp'] = prediction_timestamp
                    prediction_df_combined['prediction_host'] = prediction_host
                    prediction_df_combined['prediction_env'] = prediction_env
                    
                    prediction_df_combined['filename'] = prediction_df_combined['filename'].apply(os.path.basename)
                    
                    # Save predictions
                    result_path = os.path.join(results_folder, 'ViT_predictions.csv')
                    prediction_df_combined.to_csv(result_path, index=False)
                    logger.log(f"  Predictions saved")
                    
                    # Merge with main dataframe
                    df = pd.merge(df, prediction_df_combined, on='filename', how='left', validate='1:1')
                    
                except Exception as e:
                    logger.log(f"  ViT prediction failed: {e}")
                    logger.log(traceback.format_exc())
                    predict_ViT = False
        
        # Create EcoTaxa output directory
        ecotaxa_output_folder = os.path.join(results_folder, 'EcoTaxa')
        os.makedirs(ecotaxa_output_folder, exist_ok=True)
        
        # Prepare EcoTaxa metadata for all particles
        df_ET = ap.rename_for_ecotaxa(df, sample_profile_id=profile_info.profile_id, predicted=predict_ViT).copy()
        
        if predict_ViT:
            try:
                annotation_col = ('object_annotation_category', '[t]')
                df_ET = df_ET[df_ET[annotation_col].notna()]
                logger.log(f"  {len(df_ET)} crops with valid annotations")
            except Exception as e:
                logger.log(f"  Could not filter annotations: {e}")
        
        # ALWAYS save df_ET as TSV (EcoTaxa-ready format)
        if not df_ET.empty:
            tsv_path = os.path.join(ecotaxa_output_folder, f'{profile_name}_ecotaxa.tsv')
            df_ET.to_csv(tsv_path, sep='\t', index=False)
            logger.log(f"  EcoTaxa TSV saved: {tsv_path}")
        
        # Create ZIP files only if export_zip is True
        if export_zip:
            # Split into non-living and other classes
            if not df_ET.empty and predict_ViT:
                try:
                    annotation_col = ('object_annotation_category', '[t]')
                    
                    # Non-living particles
                    df_ET_nonliving = df_ET[df_ET[annotation_col] == 'non-living'].copy()
                    
                    # All other classes (living + unknown)
                    df_ET_other = df_ET[df_ET[annotation_col] != 'non-living'].copy()
                    
                    logger.log(f"  Non-living particles: {len(df_ET_nonliving)}")
                    logger.log(f"  Other classes (living + unknown): {len(df_ET_other)}")
                    
                    # Export non-living
                    if not df_ET_nonliving.empty:
                        try:
                            logger.log(f"  Exporting non-living particles to EcoTaxa...")
                            nonliving_folder = os.path.join(ecotaxa_output_folder, 'non-living')
                            os.makedirs(nonliving_folder, exist_ok=True)
                            
                            ap.create_ecotaxa_zips(
                                output_folder=nonliving_folder,
                                df=df_ET_nonliving,
                                profile_name=f"{profile_name}_non-living",
                                max_zip_size_mb=500,
                                compression_ratio=1.0,
                                copy_images=True,
                                add_scale_bar_to_deconv=False,
                                pixel_resolution=23,
                                scale_length_mm=1
                            )
                            logger.log(f"  Non-living export completed")
                        except Exception as e:
                            logger.log(f"  Non-living export failed: {e}")
                            logger.log(traceback.format_exc())
                    else:
                        logger.log(f"  No non-living particles to export")
                    
                    # Export other classes
                    if not df_ET_other.empty:
                        try:
                            logger.log(f"  Exporting other classes to EcoTaxa...")
                            other_folder = os.path.join(ecotaxa_output_folder, 'other')
                            os.makedirs(other_folder, exist_ok=True)
                            
                            ap.create_ecotaxa_zips(
                                output_folder=other_folder,
                                df=df_ET_other,
                                profile_name=f"{profile_name}_other",
                                max_zip_size_mb=500,
                                compression_ratio=1.0,
                                copy_images=True,
                                add_scale_bar_to_deconv=False,
                                pixel_resolution=23,
                                scale_length_mm=1
                            )
                            logger.log(f"  Other classes export completed")
                        except Exception as e:
                            logger.log(f"  Other classes export failed: {e}")
                            logger.log(traceback.format_exc())
                    else:
                        logger.log(f"  No other class particles to export")
                    
                except Exception as e:
                    logger.log(f"  Failed to split particles for separate exports: {e}")
                    logger.log(traceback.format_exc())
                    # Fallback: export all together if split fails
                    if not df_ET.empty:
                        try:
                            logger.log(f"  Exporting all particles to EcoTaxa (fallback)...")
                            ap.create_ecotaxa_zips(
                                output_folder=ecotaxa_output_folder,
                                df=df_ET,
                                profile_name=profile_name,
                                max_zip_size_mb=500,
                                compression_ratio=1.0,
                                copy_images=True,
                                add_scale_bar_to_deconv=False,
                                pixel_resolution=23,
                                scale_length_mm=1
                            )
                            logger.log(f"  All-data export completed (fallback)")
                        except Exception as e2:
                            logger.log(f"  Fallback export also failed: {e2}")
            
            elif not df_ET.empty:
                # No ViT predictions, just export all data
                try:
                    logger.log(f"  Exporting all particles to EcoTaxa...")
                    ap.create_ecotaxa_zips(
                        output_folder=ecotaxa_output_folder,
                        df=df_ET,
                        profile_name=profile_name,
                        max_zip_size_mb=500,
                        compression_ratio=1.0,
                        copy_images=True,
                        add_scale_bar_to_deconv=False,
                        pixel_resolution=23,
                        scale_length_mm=1
                    )
                    logger.log(f"  EcoTaxa export completed")
                except Exception as e:
                    logger.log(f"  EcoTaxa export failed: {e}")
                    logger.log(traceback.format_exc())
            else:
                logger.log(f"  No valid data for EcoTaxa export")
        else:
            logger.log(f"  Skipping ZIP export (disabled)")
        
        # Save metadata
        metadata_save_path = os.path.join(ecotaxa_output_folder, f'{profile_name}_crops_metadata.csv')
        df.to_csv(metadata_save_path, index=False)
        logger.log(f"  Metadata saved")
        
        return True
        
    except Exception as e:
        logger.log(f"ERROR processing {profile_name}: {e}")
        logger.log(traceback.format_exc())
        return False


def process_benchmark_mode(
    source_root: str,
    output_root: str,
    profiles_per_cruise: int = 5,
    deconvolution: bool = True,
    image_ext: str = ".png",
    run_postanalysis: bool = True,
    predict_ViT: bool = True,
    export_zip: bool = True,
    ctd_configs: Optional[Dict[str, Dict[str, str]]] = None,
    log_configs: Optional[Dict[str, str]] = None
):
    """
    Build benchmark dataset with full profiles, segment, and run post-analysis.
    This is the benchmark mode from segment_benchmark_v3.py

    Args:
        export_zip: If True, create EcoTaxa zip files. If False, only save df_ET as TSV.
        ctd_configs: Dict mapping cruise name to {"dir": path, "prefix": prefix}
        log_configs: Dict mapping cruise name to log directory path
    """
    os.makedirs(output_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_root, f"benchmark_creation_{timestamp}.log")
    logger = Logger(log_path)

    logger.log("Benchmark Dataset Creation & Analysis Log")
    logger.log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Source: {source_root}")
    logger.log(f"Output: {output_root}")
    logger.log(f"Profiles per cruise: {profiles_per_cruise}")
    logger.log(f"Deconvolution: {deconvolution}")
    logger.log(f"Post-analysis: {run_postanalysis}")
    logger.log(f"ViT predictions: {predict_ViT}")
    logger.log(f"Export ZIP: {export_zip}")
    logger.log(f"Log file: {log_path}")
    logger.log("")

    cruises = [
        d for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d))
        and d != "PISCO_KOSMOS_2020_Peru"
        and d != "M181"
        and not d.startswith(".")
    ]

    global_stats: Dict[str, Dict] = {}

    for cruise in sorted(cruises):
        cruise_path = os.path.join(source_root, cruise)

        logger.log("\n" + "=" * 70)
        logger.log(f"CRUISE: {cruise}")
        logger.log("=" * 70)

        profiles_base = os.path.join(cruise_path, f"{cruise}-PISCO-Profiles")
        if not os.path.exists(profiles_base):
            logger.log("  ERROR: No PISCO-Profiles directory found")
            continue

        profile_dirs = sorted([
            d for d in os.listdir(profiles_base)
            if os.path.isdir(os.path.join(profiles_base, d))
            and not d.startswith(".")
        ])

        if not profile_dirs:
            logger.log(f"  ERROR: No profiles found in {profiles_base}")
            continue

        logger.log(f"  Total profiles available: {len(profile_dirs)}")

        profile_image_dirs: Dict[str, List[str]] = {}
        for profile_dir in profile_dirs:
            profile_path = os.path.join(profiles_base, profile_dir)
            image_dirs = find_image_dirs(profile_path, image_ext)
            if image_dirs:
                profile_image_dirs[profile_dir] = image_dirs

        logger.log(f"  Profiles with images: {len(profile_image_dirs)}")

        if not profile_image_dirs:
            logger.log("  ERROR: No images found in any profile")
            continue

        num_profiles_to_select = min(profiles_per_cruise, len(profile_image_dirs))
        selected_profile_names = select_equally_spaced(
            sorted(profile_image_dirs.keys()),
            num_profiles_to_select
        )

        logger.log(f"  Selecting {len(selected_profile_names)} equally spaced profiles")

        total_segmented = 0
        profile_stats: Dict[str, Dict] = {}

        # Get CTD and log configs for this cruise
        ctd_dir = None
        ctd_prefix = None
        log_directory = None
        
        if ctd_configs and cruise in ctd_configs:
            ctd_dir = ctd_configs[cruise].get("dir")
            ctd_prefix = ctd_configs[cruise].get("prefix")
        
        if log_configs and cruise in log_configs:
            log_directory = log_configs[cruise]

        for profile_name in sorted(selected_profile_names):
            image_dirs = profile_image_dirs[profile_name]
            profile_output = os.path.join(output_root, profile_name)
            os.makedirs(profile_output, exist_ok=True)

            logger.log(f"    Segmenting {profile_name} from {len(image_dirs)} dir(s)")
            for image_dir in image_dirs:
                logger.log(f"      -> {image_dir}")
                run_segmenter(image_dir, profile_output, deconvolution)

            segmented_images = glob(os.path.join(profile_output, "Data", "*.csv"))
            segmented_count = len(segmented_images)

            profile_stats[profile_name] = {
                "segmented": segmented_count,
                "image_dirs": len(image_dirs),
                "cruise": cruise
            }

            logger.log(f"    OK: {profile_name} segmented images: {segmented_count}")
            total_segmented += segmented_count
            
            # Run post-analysis if requested
            if run_postanalysis and segmented_count > 0:
                profile_path = os.path.join(profiles_base, profile_name)
                success = process_profile_postanalysis(
                    profile_name=profile_name,
                    results_folder=profile_output,
                    cruise=cruise,
                    profile_path=profile_path,
                    logger=logger,
                    predict_ViT=predict_ViT,
                    export_zip=export_zip,
                    ctd_dir=ctd_dir,
                    ctd_prefix=ctd_prefix,
                    log_directory=log_directory
                )
                if success:
                    logger.log(f"    OK: Post-analysis completed for {profile_name}")
                else:
                    logger.log(f"    WARNING: Post-analysis failed for {profile_name}")

        global_stats[cruise] = {
            "total": total_segmented,
            "profiles_used": len(selected_profile_names),
            "profiles": profile_stats
        }

        logger.log(
            f"  OK: Total images from {len(selected_profile_names)} profiles: {total_segmented}"
        )

    logger.log("\n" + "=" * 70)
    logger.log("BENCHMARK DATASET CREATION AND ANALYSIS SUMMARY")
    logger.log("=" * 70)

    total_all = 0
    for cruise in sorted(global_stats.keys()):
        stats = global_stats[cruise]
        logger.log(
            f"{cruise:30s}: {stats['total']:5d} imgs ({stats['profiles_used']} profiles)"
        )
        total_all += stats["total"]

    logger.log("-" * 70)
    logger.log(f"{'TOTAL':30s}: {total_all:5d} images")
    logger.log("=" * 70)
    logger.log(f"\nCompleted: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log saved to: {log_path}")

    logger.close()


def process_cruise_mode(
    cruise_name: str,
    source_root: str,
    output_root: str,
    deconvolution: bool = True,
    image_ext: str = ".png",
    run_postanalysis: bool = True,
    predict_ViT: bool = True,
    export_zip: bool = True,
    ctd_config: Optional[Dict[str, str]] = None,
    log_directory: Optional[str] = None,
    profile_limit: Optional[int] = None
):
    """
    Process all profiles from a specific cruise.
    
    Args:
        cruise_name: Name of the cruise (e.g., "SO298", "MSM126")
        source_root: Root directory containing cruise folders
        output_root: Output directory for processed data
        deconvolution: Whether to apply deconvolution during segmentation
        image_ext: Image file extension to look for
        run_postanalysis: Whether to run post-analysis
        predict_ViT: Whether to run ViT predictions
        export_zip: If True, create EcoTaxa zip files. If False, only save df_ET as TSV.
        ctd_config: Dict with {"dir": path, "prefix": prefix} for CTD data
        log_directory: Path to log directory
        profile_limit: If set, only process this many profiles (for testing)
    """
    os.makedirs(output_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_root, f"{cruise_name}_processing_{timestamp}.log")
    logger = Logger(log_path)

    logger.log(f"Cruise Processing Log: {cruise_name}")
    logger.log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Source: {source_root}")
    logger.log(f"Output: {output_root}")
    logger.log(f"Deconvolution: {deconvolution}")
    logger.log(f"Post-analysis: {run_postanalysis}")
    logger.log(f"ViT predictions: {predict_ViT}")
    logger.log(f"Export ZIP: {export_zip}")
    if profile_limit:
        logger.log(f"Profile limit: {profile_limit}")
    logger.log(f"Log file: {log_path}")
    logger.log("")

    # Find cruise directory
    cruise_path = os.path.join(source_root, cruise_name)
    if not os.path.exists(cruise_path):
        logger.log(f"ERROR: Cruise directory not found: {cruise_path}")
        logger.close()
        return

    # Find profiles directory
    profiles_base = os.path.join(cruise_path, f"{cruise_name}-PISCO-Profiles")
    if not os.path.exists(profiles_base):
        logger.log(f"ERROR: No PISCO-Profiles directory found at {profiles_base}")
        logger.close()
        return

    # Get all profile directories
    profile_dirs = sorted([
        d for d in os.listdir(profiles_base)
        if os.path.isdir(os.path.join(profiles_base, d))
        and not d.startswith(".")
    ])

    if not profile_dirs:
        logger.log(f"ERROR: No profiles found in {profiles_base}")
        logger.close()
        return

    logger.log(f"Total profiles available: {len(profile_dirs)}")

    # Filter to profiles with images
    profile_image_dirs: Dict[str, List[str]] = {}
    for profile_dir in profile_dirs:
        profile_path = os.path.join(profiles_base, profile_dir)
        image_dirs = find_image_dirs(profile_path, image_ext)
        if image_dirs:
            profile_image_dirs[profile_dir] = image_dirs

    logger.log(f"Profiles with images: {len(profile_image_dirs)}")

    if not profile_image_dirs:
        logger.log("ERROR: No images found in any profile")
        logger.close()
        return

    # Apply profile limit if specified
    selected_profiles = sorted(profile_image_dirs.keys())
    if profile_limit:
        selected_profiles = selected_profiles[:profile_limit]
        logger.log(f"Processing limited to {len(selected_profiles)} profiles")
    else:
        logger.log(f"Processing all {len(selected_profiles)} profiles")

    # Get CTD config
    ctd_dir = None
    ctd_prefix = None
    if ctd_config:
        ctd_dir = ctd_config.get("dir")
        ctd_prefix = ctd_config.get("prefix")

    # Process each profile
    total_segmented = 0
    successful_profiles = 0
    failed_profiles = []

    for i, profile_name in enumerate(selected_profiles, 1):
        logger.log("\n" + "=" * 70)
        logger.log(f"PROFILE {i}/{len(selected_profiles)}: {profile_name}")
        logger.log("=" * 70)

        image_dirs = profile_image_dirs[profile_name]
        profile_output = os.path.join(output_root, profile_name)
        os.makedirs(profile_output, exist_ok=True)

        # Segment
        logger.log(f"  Segmenting from {len(image_dirs)} image directory(ies)")
        for image_dir in image_dirs:
            logger.log(f"    -> {image_dir}")
            try:
                run_segmenter(image_dir, profile_output, deconvolution)
            except Exception as e:
                logger.log(f"    ERROR during segmentation: {e}")
                logger.log(traceback.format_exc())

        # Count segmented images
        segmented_images = glob(os.path.join(profile_output, "Data", "*.csv"))
        segmented_count = len(segmented_images)
        logger.log(f"  Segmented: {segmented_count} images")
        total_segmented += segmented_count

        # Run post-analysis if requested
        if run_postanalysis and segmented_count > 0:
            profile_path = os.path.join(profiles_base, profile_name)
            success = process_profile_postanalysis(
                profile_name=profile_name,
                results_folder=profile_output,
                cruise=cruise_name,
                profile_path=profile_path,
                logger=logger,
                predict_ViT=predict_ViT,
                export_zip=export_zip,
                ctd_dir=ctd_dir,
                ctd_prefix=ctd_prefix,
                log_directory=log_directory
            )
            if success:
                successful_profiles += 1
                logger.log(f"  OK: Profile completed successfully")
            else:
                failed_profiles.append(profile_name)
                logger.log(f"  WARNING: Post-analysis failed")
        elif segmented_count == 0:
            failed_profiles.append(profile_name)
            logger.log(f"  WARNING: No images segmented")

    # Summary
    logger.log("\n" + "=" * 70)
    logger.log(f"CRUISE PROCESSING SUMMARY: {cruise_name}")
    logger.log("=" * 70)
    logger.log(f"Total profiles processed: {len(selected_profiles)}")
    logger.log(f"Successful profiles: {successful_profiles}")
    logger.log(f"Failed profiles: {len(failed_profiles)}")
    logger.log(f"Total segmented images: {total_segmented}")
    
    if failed_profiles:
        logger.log("\nFailed profiles:")
        for prof in failed_profiles:
            logger.log(f"  - {prof}")
    
    logger.log("=" * 70)
    logger.log(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log saved to: {log_path}")

    logger.close()


def main():
    parser = argparse.ArgumentParser(
        description="PISCO Profile Processor - Segment and analyze PISCO profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark mode (5 profiles per cruise)
  %(prog)s --mode benchmark --output /path/to/output --profiles-per-cruise 5
  
  # Process entire SO298 cruise without ZIP export
  %(prog)s --mode cruise --cruise SO298 --output /path/to/output --no-export-zip
  
  # Process MSM126 cruise with ZIP export enabled
  %(prog)s --mode cruise --cruise MSM126 --output /path/to/output --export-zip
  
  # Test run on first 3 profiles of a cruise
  %(prog)s --mode cruise --cruise M202 --output /path/to/output --profile-limit 3
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['benchmark', 'cruise'],
        help='Processing mode: benchmark (multiple cruises) or cruise (single cruise)'
    )
    
    parser.add_argument(
        '--cruise',
        type=str,
        help='Cruise name (required for cruise mode). E.g., SO298, MSM126, M202, SO308, HE570'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='/mnt/filer',
        help='Source root directory containing cruise folders (default: /mnt/filer)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output root directory for processed data'
    )
    
    # Processing options
    parser.add_argument(
        '--profiles-per-cruise',
        type=int,
        default=5,
        help='Number of profiles per cruise in benchmark mode (default: 5)'
    )
    
    parser.add_argument(
        '--profile-limit',
        type=int,
        help='Limit number of profiles to process in cruise mode (for testing)'
    )
    
    parser.add_argument(
        '--no-deconv',
        action='store_true',
        help='Disable deconvolution during segmentation'
    )
    
    parser.add_argument(
        '--no-postanalysis',
        action='store_true',
        help='Skip post-analysis (only do segmentation)'
    )
    
    parser.add_argument(
        '--no-vit',
        action='store_true',
        help='Skip ViT predictions'
    )
    
    # ZIP export toggle
    parser.add_argument(
        '--export-zip',
        action='store_true',
        default=False,
        help='Enable EcoTaxa ZIP file export (disabled by default)'
    )
    
    parser.add_argument(
        '--no-export-zip',
        action='store_true',
        default=False,
        help='Disable EcoTaxa ZIP file export (only save TSV)'
    )
    
    parser.add_argument(
        '--image-ext',
        type=str,
        default='.png',
        help='Image file extension (default: .png)'
    )
    
    args = parser.parse_args()
    
    # Validate cruise mode arguments
    if args.mode == 'cruise' and not args.cruise:
        parser.error("--cruise is required when using cruise mode")
    
    # Handle ZIP export logic
    if args.export_zip and args.no_export_zip:
        parser.error("Cannot specify both --export-zip and --no-export-zip")
    
    export_zip = args.export_zip  # Default is False unless --export-zip is specified
    
    # CTD configurations
    ctd_configs = {
        "SO298": {"dir": "/mnt/filer/SO298/SO298-CTD_UVP_ETC/SO298-CTD/calibrated/", "prefix": "son_298_1_"},
        "MSM126": {"dir": "/mnt/filer/MSM126/MSM126-Data-UVP-CTD-ADCP/CTD/msm_126_1_ctd", "prefix": "msm_126_1_"},
        "M202": {"dir": "/mnt/filer/M202/M202-external-Data/M202-CTD/met_202_1_ctd/met_202_1_ctd", "prefix": "met_202_1_"},
        "SO308": {"dir": "/mnt/filer/SO308/ADCP-ETC/CTD/SO308_ctd_files", "prefix": "son_308_1_"},
    }
    
    # Log configurations
    log_configs = {
        "SO298": "/mnt/filer/SO298/SO298-Logfiles_PISCO/Templog",
        "MSM126": "/mnt/filer/MSM126/MSM126-PISCO_Logfilesetc/Logfiles",
        "M202": "/mnt/filer/M202/M202-Pisco-Logfiles/Logfiles",
        "SO308": "/mnt/filer/SO308/PISCO-Logfiles/PISCO-LOGFILES/Logfiles",
    }
    
    # Run appropriate mode
    if args.mode == 'benchmark':
        print(f"Running in BENCHMARK mode")
        print(f"  Profiles per cruise: {args.profiles_per_cruise}")
        print(f"  Export ZIP: {export_zip}")
        print(f"  Output: {args.output}")
        
        process_benchmark_mode(
            source_root=args.source,
            output_root=args.output,
            profiles_per_cruise=args.profiles_per_cruise,
            deconvolution=not args.no_deconv,
            image_ext=args.image_ext,
            run_postanalysis=not args.no_postanalysis,
            predict_ViT=not args.no_vit,
            export_zip=export_zip,
            ctd_configs=ctd_configs,
            log_configs=log_configs
        )
    
    elif args.mode == 'cruise':
        print(f"Running in CRUISE mode")
        print(f"  Cruise: {args.cruise}")
        print(f"  Export ZIP: {export_zip}")
        print(f"  Output: {args.output}")
        if args.profile_limit:
            print(f"  Profile limit: {args.profile_limit}")
        
        process_cruise_mode(
            cruise_name=args.cruise,
            source_root=args.source,
            output_root=args.output,
            deconvolution=not args.no_deconv,
            image_ext=args.image_ext,
            run_postanalysis=not args.no_postanalysis,
            predict_ViT=not args.no_vit,
            export_zip=export_zip,
            ctd_config=ctd_configs.get(args.cruise),
            log_directory=log_configs.get(args.cruise),
            profile_limit=args.profile_limit
        )


if __name__ == "__main__":
    main()