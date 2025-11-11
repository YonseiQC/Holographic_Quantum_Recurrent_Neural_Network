import matplotlib.pyplot as plt
from hqrnn.scheduler.loss_history import LossHistory
from hqrnn.FFF_mode.types import UnifiedModeState

# --- 10. Plotting Utility

def save_loss_plot(plots_dir, config, loss_history: LossHistory, epoch: int, mode_state: UnifiedModeState):
    if not loss_history.epochs:
        return

    total_loss_history = loss_history.losses.get('total')

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Training Analysis - Epoch {epoch}', fontsize=16)

    # (0,0) Loss curves
    ax = axs[0, 0]
    if total_loss_history:
        ax.plot(loss_history.epochs, total_loss_history, color='darkslateblue', linewidth=2, label='Total Loss')

    if config.model == 3:
        ds_cfg = config.dataset_cfg
        first_digit_loss = loss_history.losses.get('first_digit_loss')
        second_digit_loss = loss_history.losses.get('second_digit_loss')
        if first_digit_loss:
            ax.plot(loss_history.epochs, first_digit_loss, color='palevioletred', alpha=0.8, label=f'Digit {ds_cfg.first_digit} Loss')
        if second_digit_loss:
            ax.plot(loss_history.epochs, second_digit_loss, color='steelblue', alpha=0.8, label=f'Digit {ds_cfg.second_digit} Loss')
        ax.legend()

    ax.set_title('Loss Over Time'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3)

    # (0,1) LR curve
    ax = axs[0, 1]
    if len(loss_history.epochs) > 1:
        ax.plot(loss_history.epochs[1:], loss_history.learning_rates[1:], color='orange', linewidth=2)
    elif loss_history.epochs:
        ax.plot(loss_history.epochs, loss_history.learning_rates, color='orange', linewidth=2)

    # (1,0) Mode timeline
    ax = axs[1, 0]
    mode_colors = {'find': 'paleturquoise', 'fight': 'lightseagreen', 'flee': 'azure', 'super': 'darkslategrey'}
    mode_labels = {'find': 'Find', 'fight': 'Fight', 'flee': 'Flee', 'super': 'Super-Fight'}
    y_pos = 0.5; bar_height = 0.6; segment_start = 0; processed_modes = []
    for i, (mode_val, epoch_val) in enumerate(zip(loss_history.modes, loss_history.epochs)):
        is_super = (mode_state.super_fight_entry_epoch != -1 and
                    epoch_val >= mode_state.super_fight_entry_epoch and
                    mode_state.super_fight_entry_epoch + 100 > epoch_val and
                    mode_val == 'fight')
        display_mode = 'super' if is_super else mode_val
        processed_modes.append(display_mode)
        if i > 0 and display_mode != processed_modes[i - 1]:
            start_epoch = loss_history.epochs[segment_start]
            end_epoch = loss_history.epochs[i - 1]
            ax.barh(y_pos, end_epoch - start_epoch + 1, left=start_epoch, height=bar_height,
                    color=mode_colors.get(processed_modes[i - 1], 'gray'), alpha=0.7, edgecolor='black')
            segment_start = i
    if loss_history.epochs:
        start_epoch = loss_history.epochs[segment_start]
        end_epoch = loss_history.epochs[-1]
        ax.barh(y_pos, end_epoch - start_epoch + 1, left=start_epoch, height=bar_height,
                color=mode_colors.get(processed_modes[-1], 'gray'), alpha=0.7, edgecolor='black')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in mode_colors.values()]
    ax.legend(handles, mode_labels.values(), loc='upper right', ncol=len(mode_labels))
    ax.set_xlabel('Epoch'); ax.set_title('Mode Timeline'); ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(True, alpha=0.3, axis='x')

    # (1,1) Stats
    ax = axs[1, 1]; ax.axis('off')
    stats_text = f"Training Statistics (Epoch {epoch}):\n"
    if total_loss_history:
        stats_text += f"{'Total Loss':<15}: {total_loss_history[-1]:.6f}\n"
    stats_text += f"{'Current Mode':<15}: {mode_state.mode.value.upper()}\n"
    if mode_state.is_super_fight:
        stats_text += f"{'Super-Fight':<15}: Active\n"
    if loss_history.learning_rates:
        stats_text += f"{'Current LR':<15}: {loss_history.learning_rates[-1]:.2e}\n"
    stats_text += f"{'Best Loss':<15}: {mode_state.best_loss:.6f}\n\n"
    stats_text += f"Fight Count: {mode_state.fight_count}\n"
    stats_text += f"Flee Count:  {mode_state.flee_count}"
    ax.text(0.01, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.5))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = plots_dir / f"training_analysis_epoch_{epoch}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training analysis plot saved to {plot_path}")
