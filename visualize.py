import plotly.graph_objects as go
from volume_profile import load_from_csv

def plot_daily_vp(df_vp, df_levels, date):

    # filtering the raw profile for the chosen date
    day = df_vp[df_vp['date'].astype(str) == date]
    levels = df_levels[df_levels['date'].astype(str) == date].iloc[0]

    poc = levels['poc']
    vah = levels['vah']
    val = levels['val']

    # coloring each price bucket based on where it sits
    colors = []
    for price in day['price_bucket']:
        if price == poc:
            colors.append('#FF4136')       # red for POC
        elif val <= price <= vah:
            colors.append('#4DA6FF')       # blue for value area
        else:
            colors.append('#B0B0B0')       # grey outside value area

    fig = go.Figure()

    # horizontal bars — price on y axis, volume on x axis
    fig.add_trace(go.Bar(
        x=day['volume'],
        y=day['price_bucket'],
        orientation='h',
        marker_color=colors,
        opacity=0.85,
        name='Volume',
        hovertemplate='Price: %{y}<br>Volume: %{x:.3f} BTC<extra></extra>'
    ))

    # POC line
    fig.add_hline(
        y=poc,
        line_color='#FF4136',
        line_dash='dash',
        line_width=1.5,
        annotation_text=f'POC: {poc}',
        annotation_position='right',
        annotation_font_color='#FF4136'
    )

    # VAH line
    fig.add_hline(
        y=vah,
        line_color='#2ECC40',
        line_dash='dash',
        line_width=1.5,
        annotation_text=f'VAH: {vah}',
        annotation_position='right',
        annotation_font_color='#2ECC40'
    )

    # VAL line
    fig.add_hline(
        y=val,
        line_color='#FF851B',
        line_dash='dash',
        line_width=1.5,
        annotation_text=f'VAL: {val}',
        annotation_position='right',
        annotation_font_color='#FF851B'
    )

    fig.update_layout(
        title=f'Volume Profile — {date}',
        xaxis_title='Volume (BTC)',
        yaxis_title='Price (USDT)',
        plot_bgcolor='#0f0f0f',
        paper_bgcolor='#0f0f0f',
        font_color='#e2e8f0',
        height=700,
        width=800,
        showlegend=False,
        xaxis=dict(gridcolor='#1e2330'),
        yaxis=dict(gridcolor='#1e2330')
    )

    fig.show()

    print(f"\nPOC: {poc}")
    print(f"VAH: {vah}")
    print(f"VAL: {val}")
    print(f"Value Area Width: {vah - val}")


if __name__ == "__main__":
    df_vp, df_levels = load_from_csv()
    plot_daily_vp(df_vp, df_levels, '2026-01-30')