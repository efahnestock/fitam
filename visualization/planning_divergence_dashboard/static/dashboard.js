// State
let approach1Data = null;
let approach2Data = null;
let aggregateData = null;

// DOM elements
const approach1Select = document.getElementById('approach1');
const approach2Select = document.getElementById('approach2');
const trialSelect = document.getElementById('trial');
const slider1 = document.getElementById('slider1');
const slider2 = document.getElementById('slider2');
const slider1Value = document.getElementById('slider1-value');
const slider2Value = document.getElementById('slider2-value');
const costmap1 = document.getElementById('costmap1');
const costmap2 = document.getElementById('costmap2');
const approach1Label = document.getElementById('approach1-label');
const approach2Label = document.getElementById('approach2-label');

// Initialize
async function init() {
    await loadApproaches();
    setupEventListeners();
}

async function loadApproaches() {
    const response = await fetch('/api/approaches');
    const data = await response.json();

    approach1Select.innerHTML = '';
    approach2Select.innerHTML = '';

    data.approaches.forEach(approach => {
        approach1Select.add(new Option(approach, approach));
        approach2Select.add(new Option(approach, approach));
    });

    // Set defaults if available
    if (data.approaches.includes('baseline')) {
        approach1Select.value = 'baseline';
    }
    if (data.approaches.includes('diffusion')) {
        approach2Select.value = 'diffusion';
    } else if (data.approaches.includes('perfect_vision')) {
        approach2Select.value = 'perfect_vision';
    }

    await loadTrials();
}

async function loadTrials() {
    const approach1 = approach1Select.value;
    const approach2 = approach2Select.value;

    if (!approach1 || !approach2) return;

    const response = await fetch(`/api/trials?approach1=${approach1}&approach2=${approach2}`);
    const data = await response.json();

    trialSelect.innerHTML = '';
    data.trials.forEach(trial => {
        trialSelect.add(new Option(trial, trial));
    });

    // Load aggregate data for the scatter plot
    await loadAggregateData();

    if (data.trials.length > 0) {
        await loadTrialData();
    } else {
        clearPlot();
        clearImages();
    }
}

async function loadAggregateData() {
    const approach1 = approach1Select.value;
    const approach2 = approach2Select.value;

    if (!approach1 || !approach2) return;

    const response = await fetch(`/api/aggregate-data?approach1=${approach1}&approach2=${approach2}`);
    const data = await response.json();
    aggregateData = data.data;

    plotAggregate();
}

function plotAggregate() {
    if (!aggregateData || aggregateData.length === 0) {
        Plotly.purge('aggregate-plot');
        return;
    }

    const x = aggregateData.map(d => d.cost1);
    const y = aggregateData.map(d => d.cost2);
    const trials = aggregateData.map(d => d.trial);

    const maxVal = Math.max(...x, ...y);

    const trace = {
        x: x,
        y: y,
        text: trials,
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: 10,
            color: '#1f77b4',
        },
        hovertemplate: '%{text}<br>%{x:.1f} vs %{y:.1f}<extra></extra>',
    };

    // Diagonal reference line
    const diagonalLine = {
        x: [0, maxVal],
        y: [0, maxVal],
        mode: 'lines',
        type: 'scatter',
        line: {
            color: 'gray',
            width: 1,
            dash: 'dot',
        },
        hoverinfo: 'skip',
        showlegend: false,
    };

    const layout = {
        title: 'Aggregate: Final Cost Comparison',
        xaxis: {
            title: approach1Select.value,
            scaleanchor: 'y',
            scaleratio: 1,
            rangemode: 'tozero',
        },
        yaxis: {
            title: approach2Select.value,
            rangemode: 'tozero',
        },
        hovermode: 'closest',
        showlegend: false,
    };

    Plotly.newPlot('aggregate-plot', [diagonalLine, trace], layout);

    // Click handler to navigate to trial
    document.getElementById('aggregate-plot').on('plotly_click', function(data) {
        const pointIndex = data.points[0].pointIndex;
        // Account for diagonal line being trace 0
        if (data.points[0].curveNumber === 1) {
            const trial = aggregateData[pointIndex].trial;
            trialSelect.value = trial;
            loadTrialData();
        }
    });
}

function updateAggregateHighlight() {
    if (!aggregateData || aggregateData.length === 0) return;

    const currentTrial = trialSelect.value;
    const colors = aggregateData.map(d =>
        d.trial === currentTrial ? '#ff7f0e' : '#1f77b4'
    );
    const sizes = aggregateData.map(d =>
        d.trial === currentTrial ? 14 : 10
    );

    Plotly.restyle('aggregate-plot', {
        'marker.color': [colors],
        'marker.size': [sizes],
    }, [1]);  // Update trace 1 (the scatter points)
}

async function loadTrialData() {
    const approach1 = approach1Select.value;
    const approach2 = approach2Select.value;
    const trial = trialSelect.value;

    if (!approach1 || !approach2 || !trial) return;

    // Update labels
    approach1Label.textContent = approach1;
    approach2Label.textContent = approach2;

    // Load data for both approaches in parallel
    const [resp1, resp2] = await Promise.all([
        fetch(`/api/trial-data?approach=${approach1}&trial=${trial}`),
        fetch(`/api/trial-data?approach=${approach2}&trial=${trial}`)
    ]);

    if (!resp1.ok || !resp2.ok) {
        console.error('Failed to load trial data');
        return;
    }

    approach1Data = await resp1.json();
    approach2Data = await resp2.json();

    // Update sliders
    slider1.max = Math.max(0, approach1Data.costmap_count - 1);
    slider2.max = Math.max(0, approach2Data.costmap_count - 1);
    slider1.value = 0;
    slider2.value = 0;
    updateSliderLabels();

    // Plot costs
    plotCosts();

    // Highlight current trial in aggregate plot
    updateAggregateHighlight();

    // Load initial images
    loadImages();
}

function plotCosts() {
    if (!approach1Data || !approach2Data) return;

    // accumulated_cost is the cumulative estimated cost at each planning iteration
    const accumulated1 = approach1Data.accumulated_cost;
    const accumulated2 = approach2Data.accumulated_cost;

    const trace1 = {
        y: accumulated1,
        x: Array.from({ length: accumulated1.length }, (_, i) => i),
        name: approach1Select.value,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#1f77b4', width: 2 }
    };

    const trace2 = {
        y: accumulated2,
        x: Array.from({ length: accumulated2.length }, (_, i) => i),
        name: approach2Select.value,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ff7f0e', width: 2 }
    };

    // Add vertical lines for current slider positions
    const shapes = [
        {
            type: 'line',
            x0: parseInt(slider1.value),
            x1: parseInt(slider1.value),
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#1f77b4', width: 2, dash: 'dash' }
        },
        {
            type: 'line',
            x0: parseInt(slider2.value),
            x1: parseInt(slider2.value),
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#ff7f0e', width: 2, dash: 'dash' }
        }
    ];

    const layout = {
        title: 'Accumulated Estimated Cost',
        xaxis: { title: 'Planning Iteration' },
        yaxis: { title: 'Accumulated Cost' },
        shapes: shapes,
        hovermode: 'closest',
        legend: { x: 0, y: 1 }
    };

    Plotly.newPlot('cost-plot', [trace1, trace2], layout);

    // Add click handler - update both sliders on any click
    document.getElementById('cost-plot').on('plotly_click', function(data) {
        const clickedX = Math.round(data.points[0].x);

        slider1.value = Math.min(clickedX, slider1.max);
        slider2.value = Math.min(clickedX, slider2.max);

        updateSliderLabels();
        loadImages();
        updatePlotMarkers();
    });
}

function updatePlotMarkers() {
    const shapes = [
        {
            type: 'line',
            x0: parseInt(slider1.value),
            x1: parseInt(slider1.value),
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#1f77b4', width: 2, dash: 'dash' }
        },
        {
            type: 'line',
            x0: parseInt(slider2.value),
            x1: parseInt(slider2.value),
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: '#ff7f0e', width: 2, dash: 'dash' }
        }
    ];

    Plotly.relayout('cost-plot', { shapes: shapes });
}

function updateSliderLabels() {
    slider1Value.textContent = `${slider1.value} / ${slider1.max}`;
    slider2Value.textContent = `${slider2.value} / ${slider2.max}`;
}

let imageLoadTimeout = null;

function loadImages() {
    // Debounce image loading
    if (imageLoadTimeout) {
        clearTimeout(imageLoadTimeout);
    }

    imageLoadTimeout = setTimeout(() => {
        const approach1 = approach1Select.value;
        const approach2 = approach2Select.value;
        const trial = trialSelect.value;

        if (!approach1 || !approach2 || !trial) return;

        costmap1.classList.add('loading');
        costmap2.classList.add('loading');

        costmap1.src = `/api/costmap-image?approach=${approach1}&trial=${trial}&index=${slider1.value}`;
        costmap2.src = `/api/costmap-image?approach=${approach2}&trial=${trial}&index=${slider2.value}`;
    }, 50);
}

function clearPlot() {
    Plotly.purge('cost-plot');
}

function clearImages() {
    costmap1.src = '';
    costmap2.src = '';
}

function setupEventListeners() {
    approach1Select.addEventListener('change', loadTrials);
    approach2Select.addEventListener('change', loadTrials);
    trialSelect.addEventListener('change', loadTrialData);

    slider1.addEventListener('input', () => {
        updateSliderLabels();
        loadImages();
        updatePlotMarkers();
    });

    slider2.addEventListener('input', () => {
        updateSliderLabels();
        loadImages();
        updatePlotMarkers();
    });

    // Image load handlers
    costmap1.addEventListener('load', () => costmap1.classList.remove('loading'));
    costmap2.addEventListener('load', () => costmap2.classList.remove('loading'));

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        // Ignore if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        const step = e.shiftKey ? 10 : 1;

        switch (e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                slider1.value = Math.max(0, parseInt(slider1.value) - step);
                slider2.value = Math.max(0, parseInt(slider2.value) - step);
                updateSliderLabels();
                loadImages();
                updatePlotMarkers();
                break;
            case 'ArrowRight':
                e.preventDefault();
                slider1.value = Math.min(parseInt(slider1.max), parseInt(slider1.value) + step);
                slider2.value = Math.min(parseInt(slider2.max), parseInt(slider2.value) + step);
                updateSliderLabels();
                loadImages();
                updatePlotMarkers();
                break;
        }
    });
}

// Start
init();
