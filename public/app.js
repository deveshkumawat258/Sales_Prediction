/**
 * SmartAd App Logic
 * Author: Harsh (Junior Developer)
 * Date: July 2026
 * Handles interactive controls, AJAX fetches to FastAPI, and Chart.js integrations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Predictor
    const tvRange = document.getElementById("tv-range");
    const tvNum = document.getElementById("tv-num");
    const radioRange = document.getElementById("radio-range");
    const radioNum = document.getElementById("radio-num");
    const newspaperRange = document.getElementById("newspaper-range");
    const newspaperNum = document.getElementById("newspaper-num");
    const liveSalesVal = document.getElementById("live-sales-val");

    // DOM Elements - Optimizer
    const budgetInput = document.getElementById("budget-input");
    const optimizeBtn = document.getElementById("optimize-btn");
    const optResults = document.getElementById("optimization-results");
    const optEmptyState = document.getElementById("optimizer-empty-state");
    
    const optSalesVal = document.getElementById("opt-sales-val");
    const optSpendVal = document.getElementById("opt-spend-val");
    const optTvVal = document.getElementById("opt-tv-val");
    const optTvPct = document.getElementById("opt-tv-pct");
    const optRadioVal = document.getElementById("opt-radio-val");
    const optRadioPct = document.getElementById("opt-radio-pct");
    const optNewsVal = document.getElementById("opt-news-val");
    const optNewsPct = document.getElementById("opt-news-pct");
    const applyOptBtn = document.getElementById("apply-opt-btn");

    // Global variables for charts
    let allocationChart = null;
    let attributionChart = null;

    // Track active optimized allocations to apply them back to sliders
    let currentOptAllocation = null;

    // Helper: update background track fill on sliders
    function updateSliderFill(slider, maxVal, color) {
        const percentage = (slider.value / maxVal) * 100;
        slider.style.background = `linear-gradient(to right, ${color} 0%, ${percentage}%, rgba(255, 255, 255, 0.1) ${percentage}%, rgba(255, 255, 255, 0.1) 100%)`;
    }

    // Connect slider inputs with number inputs
    function setupInputSync(slider, numberInput, maxVal, color, updateCallback) {
        // Sync number input to slider
        slider.addEventListener("input", (e) => {
            numberInput.value = parseFloat(e.target.value).toFixed(1);
            updateSliderFill(slider, maxVal, color);
            updateCallback();
        });

        // Sync slider to number input
        numberInput.addEventListener("change", (e) => {
            let val = parseFloat(e.target.value);
            if (isNaN(val)) val = 0;
            if (val < 0) val = 0;
            if (val > maxVal) val = maxVal;
            
            numberInput.value = val.toFixed(1);
            slider.value = val;
            updateSliderFill(slider, maxVal, color);
            updateCallback();
        });

        // Initial fill setup
        updateSliderFill(slider, maxVal, color);
    }

    // Real-time AJAX call to FastAPI /api/predict
    async function updateLivePrediction() {
        const tv = parseFloat(tvNum.value) || 0;
        const radio = parseFloat(radioNum.value) || 0;
        const newspaper = parseFloat(newspaperNum.value) || 0;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ tv, radio, newspaper })
            });

            if (response.ok) {
                const data = await response.json();
                liveSalesVal.textContent = data.predicted_sales_units.toFixed(2);
            } else {
                console.error("Failed to fetch prediction");
            }
        } catch (error) {
            console.error("Error making prediction request:", error);
        }
    }

    // Call /api/optimize
    async function runOptimization() {
        const totalBudget = parseFloat(budgetInput.value) || 0;
        if (totalBudget <= 0) {
            alert("Please enter a valid budget greater than 0");
            return;
        }

        optimizeBtn.textContent = "Calculating...";
        optimizeBtn.disabled = true;

        try {
            const response = await fetch("/api/optimize", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ total_budget: totalBudget })
            });

            if (response.ok) {
                const data = await response.json();
                renderOptimizationResults(data);
            } else {
                console.error("Optimization failed");
                alert("Failed to compute budget optimization. Please try again.");
            }
        } catch (error) {
            console.error("Error making optimization request:", error);
        } finally {
            optimizeBtn.textContent = "Optimize Allocation";
            optimizeBtn.disabled = false;
        }
    }

    // Render results on the right panel
    function renderOptimizationResults(data) {
        // Hide empty state, show results
        optEmptyState.classList.add("hidden");
        optResults.classList.remove("hidden");

        const alloc = data.ensemble_model_allocation; // Using best model (Gradient Boosting)
        currentOptAllocation = alloc;

        // Populate metrics
        optSalesVal.textContent = alloc.predicted_sales.toFixed(2);
        optSpendVal.textContent = `$${alloc.total_spend.toFixed(1)}k`;

        // Channel values
        optTvVal.textContent = `$${alloc.tv.toFixed(2)}k`;
        optRadioVal.textContent = `$${alloc.radio.toFixed(2)}k`;
        optNewsVal.textContent = `$${alloc.newspaper.toFixed(2)}k`;

        // Calculate percentages
        const total = alloc.tv + alloc.radio + alloc.newspaper;
        const tvPctVal = total > 0 ? Math.round((alloc.tv / total) * 100) : 0;
        const radioPctVal = total > 0 ? Math.round((alloc.radio / total) * 100) : 0;
        const newsPctVal = total > 0 ? Math.round((alloc.newspaper / total) * 100) : 0;

        optTvPct.textContent = `${tvPctVal}%`;
        optRadioPct.textContent = `${radioPctVal}%`;
        optNewsPct.textContent = `${newsPctVal}%`;

        // Render/update Doughnut Chart
        renderDoughnutChart([alloc.tv, alloc.radio, alloc.newspaper]);
    }

    // Helper: build/update Doughnut allocation chart
    function renderDoughnutChart(shares) {
        const ctx = document.getElementById("allocationChart").getContext("2d");
        
        if (allocationChart) {
            // Update existing chart
            allocationChart.data.datasets[0].data = shares;
            allocationChart.update();
        } else {
            // Create new chart
            allocationChart = new Chart(ctx, {
                type: "doughnut",
                data: {
                    labels: ["TV", "Radio", "Newspaper"],
                    datasets: [{
                        data: shares,
                        backgroundColor: ["#ff79c6", "#8be9fd", "#50fa7b"],
                        borderColor: "#1e1e24",
                        borderWidth: 2,
                        hoverOffset: 6
                    }]
                },
                options: {
                    cutout: "70%",
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false // Display legend in HTML breakdown
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return ` $${context.raw.toFixed(2)}k`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    // Apply optimization back to sliders
    applyOptBtn.addEventListener("click", () => {
        if (!currentOptAllocation) return;

        tvNum.value = currentOptAllocation.tv.toFixed(1);
        tvRange.value = currentOptAllocation.tv;
        updateSliderFill(tvRange, 300, "#ff79c6");

        radioNum.value = currentOptAllocation.radio.toFixed(1);
        radioRange.value = currentOptAllocation.radio;
        updateSliderFill(radioRange, 50, "#8be9fd");

        newspaperNum.value = currentOptAllocation.newspaper.toFixed(1);
        newspaperRange.value = currentOptAllocation.newspaper;
        updateSliderFill(newspaperRange, 120, "#50fa7b");

        updateLivePrediction();

        // Highlight the change with a brief glow in the predictor section
        const predictorCard = document.querySelector(".predictor-card");
        predictorCard.style.borderColor = "var(--color-sales)";
        predictorCard.style.boxShadow = "0 0 25px rgba(189, 147, 249, 0.4)";
        setTimeout(() => {
            predictorCard.style.borderColor = "";
            predictorCard.style.boxShadow = "";
        }, 1200);
    });

    // Populate Analytics Section on load
    async function loadInsights() {
        try {
            const response = await fetch("/api/insights");
            if (response.ok) {
                const data = await response.json();
                renderAttributionChart(data.ensemble_importances);
            }
        } catch (error) {
            console.error("Error loading insights:", error);
        }
    }

    // Render Attribution Bar Chart
    function renderAttributionChart(importances) {
        const ctx = document.getElementById("attributionChart").getContext("2d");
        
        const labels = Object.keys(importances);
        const values = Object.values(importances).map(v => v * 100); // convert to percentages

        attributionChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Relative Attribution Weight (%)",
                    data: values,
                    backgroundColor: ["#ff79c6", "#8be9fd", "#50fa7b"],
                    borderRadius: 8,
                    maxBarThickness: 50
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal bars
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: "#252530"
                        },
                        ticks: {
                            color: "#9aa0a6",
                            font: {
                                family: "Outfit"
                            },
                            callback: function(value) {
                                return value + "%";
                            }
                        },
                        max: 100
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: "#ffffff",
                            font: {
                                family: "Outfit",
                                weight: "bold",
                                size: 13
                            }
                        }
                    }
                }
            }
        });
    }

    // Set up Input synchronizations & listeners
    setupInputSync(tvRange, tvNum, 300, "#ff79c6", updateLivePrediction);
    setupInputSync(radioRange, radioNum, 50, "#8be9fd", updateLivePrediction);
    setupInputSync(newspaperRange, newspaperNum, 120, "#50fa7b", updateLivePrediction);

    // Initial predictions & insights on load
    updateLivePrediction();
    loadInsights();

    // Event listener for optimize button
    optimizeBtn.addEventListener("click", runOptimization);
    budgetInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            runOptimization();
        }
    });
});
