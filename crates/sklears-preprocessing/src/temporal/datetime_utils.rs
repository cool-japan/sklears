//! DateTime utilities and date/time component extraction
//!
//! This module provides utilities for working with timestamps and extracting
//! various date and time components for temporal feature engineering.

use sklears_core::types::Float;

/// Simple DateTime structure for working with Unix timestamps
#[derive(Debug, Clone, Copy)]
pub struct DateTime {
    /// Unix timestamp in seconds since epoch
    pub timestamp: i64,
}

impl DateTime {
    /// Create a new DateTime from Unix timestamp
    pub fn from_timestamp(timestamp: i64) -> Self {
        Self { timestamp }
    }

    /// Convert timestamp to date components assuming UTC
    /// This is a simplified implementation - in practice you'd use a proper datetime library
    pub fn to_components(&self, timezone_offset: Option<Float>) -> DateComponents {
        let mut timestamp = self.timestamp;

        // Apply timezone offset if provided
        if let Some(offset) = timezone_offset {
            timestamp += (offset * 3600.0) as i64;
        }

        // Simplified calculation - assumes Gregorian calendar
        let days_since_epoch = timestamp / 86400;
        let seconds_in_day = timestamp % 86400;

        // Calculate year (simplified)
        let mut year = 1970;
        let mut remaining_days = days_since_epoch;

        // This is a very simplified year calculation
        while remaining_days >= 365 {
            let days_in_year = if Self::is_leap_year(year) { 366 } else { 365 };
            if remaining_days >= days_in_year {
                remaining_days -= days_in_year;
                year += 1;
            } else {
                break;
            }
        }

        // Calculate month and day (simplified)
        let days_in_months = if Self::is_leap_year(year) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };

        let mut month = 1;
        let mut day_of_month = remaining_days + 1;

        for &days_in_month in &days_in_months {
            if day_of_month > days_in_month {
                day_of_month -= days_in_month;
                month += 1;
            } else {
                break;
            }
        }

        // Calculate time components
        let hour = (seconds_in_day / 3600) as u8;
        let minute = ((seconds_in_day % 3600) / 60) as u8;
        let second = (seconds_in_day % 60) as u8;

        // Calculate additional components
        let day_of_week = ((days_since_epoch + 4) % 7) as u8; // Jan 1, 1970 was Thursday (4)
        let quarter = ((month - 1) / 3 + 1) as u8;
        let day_of_year = Self::calculate_day_of_year(year, month as u8, day_of_month as u8);
        let week_of_year = Self::calculate_week_of_year(year, month as u8, day_of_month as u8);

        DateComponents {
            year: year as u32,
            month: month as u8,
            day: day_of_month as u8,
            hour,
            minute,
            second,
            day_of_week,
            quarter,
            day_of_year,
            week_of_year,
        }
    }

    fn is_leap_year(year: i64) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    fn calculate_day_of_year(year: i64, month: u8, day: u8) -> u16 {
        let days_in_months = if Self::is_leap_year(year) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };

        let mut day_of_year = day as u16;
        for i in 0..(month - 1) {
            day_of_year += days_in_months[i as usize] as u16;
        }
        day_of_year
    }

    fn calculate_week_of_year(year: i64, month: u8, day: u8) -> u8 {
        let day_of_year = Self::calculate_day_of_year(year, month, day);
        // Simplified week calculation (assumes week starts on Monday)
        ((day_of_year - 1) / 7 + 1) as u8
    }
}

/// Date and time components
#[derive(Debug, Clone, Copy)]
pub struct DateComponents {
    pub year: u32,
    pub month: u8,        // 1-12
    pub day: u8,          // 1-31
    pub hour: u8,         // 0-23
    pub minute: u8,       // 0-59
    pub second: u8,       // 0-59
    pub day_of_week: u8,  // 0-6 (Monday-Sunday)
    pub quarter: u8,      // 1-4
    pub day_of_year: u16, // 1-366
    pub week_of_year: u8, // 1-53
}
