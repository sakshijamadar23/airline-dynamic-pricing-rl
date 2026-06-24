# dashboard.py Week 3
# REAL dashboard with actual results!

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json

# Dark theme
plt.style.use('dark_background')

COLORS = {
    'Fixed Price' : '#FF6B6B',
    'Discount'    : '#FFD93D',
    'Q-Learning'  : '#6BCB77',
    'DQN'         : '#4D96FF'
}

def smooth(values, window=50):
    return np.convolve(
        values,
        np.ones(window)/window,
        mode='valid'
    )

def create_dashboard():

    # Load REAL results from Sakshi's training
    with open('results.json', 'r') as f:
        results = json.load(f)

    print("Results loaded! ✅")
    print("Building dashboard...")

    # Big figure
    fig = plt.figure(figsize=(20, 14),
                     facecolor='#0D1117')

    fig.suptitle(
        '🛫 Airline Dynamic Pricing — AI Results',
        fontsize=22,
        fontweight='bold',
        color='white',
        y=0.98
    )

    gs = gridspec.GridSpec(2, 2,
                           figure=fig,
                           hspace=0.4,
                           wspace=0.35)

    # ════════════════════════════
    # CHART 1: Learning Curves
    # ════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161B22')
    ax1.set_title(
        '📈 How AI Gets Smarter Over Time',
        fontsize=15,
        color='white'
    )

    for name, revenues in results.items():
        smoothed = smooth(revenues)
        ax1.plot(smoothed,
                label=name,
                color=COLORS[name],
                linewidth=2.5)

    ax1.set_xlabel('Episode', color='#8B949E')
    ax1.set_ylabel('Revenue ₹', color='#8B949E')
    ax1.legend(facecolor='#21262D',
               labelcolor='white',
               fontsize=12)
    ax1.tick_params(colors='#8B949E')
    ax1.grid(alpha=0.15)

    # ════════════════════════════
    # CHART 2: Revenue Comparison
    # ════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#161B22')
    ax2.set_title(
        '🏆 Who Earns Most Revenue?',
        fontsize=13,
        color='white'
    )

    names = list(results.keys())
    avgs  = [np.mean(results[n][-100:])
             for n in names]
    colors = [COLORS[n] for n in names]

    bars = ax2.bar(names, avgs,
                   color=colors,
                   width=0.5,
                   edgecolor='white')

    for bar, val in zip(bars, avgs):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200,
            f'₹{val:,.0f}',
            ha='center',
            color='white',
            fontsize=10,
            fontweight='bold'
        )

    ax2.set_ylabel('Avg Revenue ₹',
                   color='#8B949E')
    ax2.tick_params(colors='#8B949E')
    ax2.grid(axis='y', alpha=0.15)

    # ════════════════════════════
    # CHART 3: Improvement Chart
    # ════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#161B22')
    ax3.set_title(
        '📊 How Much Each Agent Improved?',
        fontsize=13,
        color='white'
    )

    improvements = []
    for name in names:
        first = np.mean(results[name][:100])
        last  = np.mean(results[name][-100:])
        improvement = ((last - first) / first) * 100
        improvements.append(improvement)

    bars2 = ax3.bar(names,
                    improvements,
                    color=colors,
                    edgecolor='white')

    for bar, val in zip(bars2, improvements):
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'+{val:.1f}%',
            ha='center',
            color='white',
            fontsize=10
        )

    ax3.set_ylabel('Improvement %',
                   color='#8B949E')
    ax3.tick_params(colors='#8B949E')
    ax3.grid(axis='y', alpha=0.15)

    # Save dashboard
    plt.savefig('dashboard.png',
                dpi=150,
                bbox_inches='tight',
                facecolor='#0D1117')
    plt.show()
    print("Dashboard saved! ✅")

if __name__ == "__main__":
    create_dashboard()