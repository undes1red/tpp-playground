def stable_palette(labels):
    # Predefined palette.
    colors = ['#4974a5', '#ffa500', '#5d782e', '#545454', '#d7b4ae', '#00ff00', '#00ffff', '#6fa287', \
              '#ee30a7', '#9e482f', '#8b02e7', '#141387', '#eccb00', '#ff0000', '#ff9664', '#b73e64', \
              '#0000ff', '#969696', '#969600', '#ffe600', '#ff6400', '#1f1e33']
    
    assert len(labels) < 23, 'Too many labels.'

    return {label: color for (label, color) in zip(labels, colors)}