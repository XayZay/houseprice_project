/**
 * House Price Prediction System - JavaScript
 * Handles form submission, API calls, and UI interactions
 */

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultContainer = document.getElementById('resultContainer');
const errorContainer = document.getElementById('errorContainer');
const loadingContainer = document.getElementById('loadingContainer');
const predictedPriceElement = document.getElementById('predictedPrice');
const errorMessageElement = document.getElementById('errorMessage');
const neighborhoodSelect = document.getElementById('neighborhood');

// Metric Elements
const maeElement = document.getElementById('mae');
const rmseElement = document.getElementById('rmse');
const r2Element = document.getElementById('r2');

/**
 * Initialize the page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    loadNeighborhoods();
    loadMetrics();
});

/**
 * Load neighborhoods from API
 */
function loadNeighborhoods() {
    fetch('/neighborhoods')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load neighborhoods');
            }
            return response.json();
        })
        .then(data => {
            if (data.neighborhoods && Array.isArray(data.neighborhoods)) {
                populateNeighborhoodSelect(data.neighborhoods);
            }
        })
        .catch(error => {
            console.error('Error loading neighborhoods:', error);
            showError('Failed to load neighborhoods. Please refresh the page.');
        });
}

/**
 * Populate neighborhood dropdown
 */
function populateNeighborhoodSelect(neighborhoods) {
    neighborhoods.forEach(neighborhood => {
        const option = document.createElement('option');
        option.value = neighborhood;
        option.textContent = neighborhood;
        neighborhoodSelect.appendChild(option);
    });
}

/**
 * Load model metrics from API
 */
function loadMetrics() {
    fetch('/metrics')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load metrics');
            }
            return response.json();
        })
        .then(data => {
            displayMetrics(data);
        })
        .catch(error => {
            console.error('Error loading metrics:', error);
            // Don't show error to user, just fail silently
        });
}

/**
 * Display metrics on the page
 */
function displayMetrics(metrics) {
    if (metrics.MAE !== undefined) {
        maeElement.textContent = `$${parseFloat(metrics.MAE).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
    }
    if (metrics.RMSE !== undefined) {
        rmseElement.textContent = `$${parseFloat(metrics.RMSE).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
    }
    if (metrics.R2 !== undefined) {
        r2Element.textContent = parseFloat(metrics.R2).toFixed(4);
    }
}

/**
 * Handle form submission
 */
predictionForm.addEventListener('submit', function(e) {
    e.preventDefault();
    submitPrediction();
});

/**
 * Submit prediction request
 */
function submitPrediction() {
    // Get form values
    const overall_qual = document.getElementById('overall_qual').value;
    const gr_liv_area = document.getElementById('gr_liv_area').value;
    const total_bsmt_sf = document.getElementById('total_bsmt_sf').value;
    const garage_cars = document.getElementById('garage_cars').value;
    const year_built = document.getElementById('year_built').value;
    const neighborhood = document.getElementById('neighborhood').value;

    // Validate inputs
    if (!validateInputs(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, year_built, neighborhood)) {
        return;
    }

    // Show loading state
    showLoading();

    // Prepare request data
    const requestData = {
        overall_qual: parseFloat(overall_qual),
        gr_liv_area: parseFloat(gr_liv_area),
        total_bsmt_sf: parseFloat(total_bsmt_sf),
        garage_cars: parseFloat(garage_cars),
        year_built: parseInt(year_built),
        neighborhood: neighborhood
    };

    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Prediction failed');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayResult(data.formatted_price);
            } else {
                showError(data.error || 'Prediction failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'An error occurred during prediction. Please try again.');
        });
}

/**
 * Validate form inputs
 */
function validateInputs(qual, area, basement, garage, year, neighborhood) {
    if (!qual || !area || !basement || !garage || !year || !neighborhood) {
        showError('All fields are required.');
        return false;
    }

    const qualNum = parseFloat(qual);
    const areaNum = parseFloat(area);
    const basementNum = parseFloat(basement);
    const garageNum = parseFloat(garage);
    const yearNum = parseInt(year);

    if (qualNum < 1 || qualNum > 10) {
        showError('Overall Quality must be between 1 and 10.');
        return false;
    }

    if (areaNum < 0) {
        showError('Living Area must be positive.');
        return false;
    }

    if (basementNum < 0) {
        showError('Basement Area must be positive.');
        return false;
    }

    if (garageNum < 0 || garageNum > 5) {
        showError('Garage Cars must be between 0 and 5.');
        return false;
    }

    if (yearNum < 1800 || yearNum > 2024) {
        showError('Year Built must be between 1800 and 2024.');
        return false;
    }

    return true;
}

/**
 * Display prediction result
 */
function displayResult(price) {
    hideLoading();
    hideError();
    predictedPriceElement.textContent = price;
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show error message
 */
function showError(message) {
    hideLoading();
    errorMessageElement.textContent = message;
    errorContainer.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    errorContainer.classList.add('hidden');
}

/**
 * Show loading state
 */
function showLoading() {
    hideError();
    resultContainer.classList.add('hidden');
    loadingContainer.classList.remove('hidden');
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingContainer.classList.add('hidden');
}

/**
 * Format currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

// Additional utility: Auto-hide error/result after 10 seconds (optional)
document.addEventListener('click', function(e) {
    // Allow clicking within form inputs
    if (e.target.closest('#predictionForm')) {
        hideError();
    }
});
