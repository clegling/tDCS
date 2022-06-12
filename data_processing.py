import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from eegtools.io import edfplus
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import zscore
from sklearn.decomposition import FastICA, PCA
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display

class StimualtionMarker(object):
    ANODAL = 'anodal'
    CATHODAL = 'cathodal'
    SHAM = 'sham'

prefix = {}
prefix[StimualtionMarker.SHAM] = "Коростелев/09042017"
prefix[StimualtionMarker.ANODAL] = "Коростелев/16042017"
prefix[StimualtionMarker.CATHODAL] = "Коростелев/23042017"


class StageMarker(object):
    BEFORE_STIMULATION = 'before'
    AFTER_STIMULATION = 'after'
    LETTER_CANCELLATION = 'letter'
    ARITHMETICS = 'arithmetic'
    AT_THE_END_OF_EXP = 'end'


class AnnotationMarker(object):
    OPENED_EYES_MARKER = 'OG'
    CLOSED_EYES_MARKER = 'ZG'
    PROBE_MARKER = 'PP'


class TimeMarker(object):
    START = 'start'
    END = 'end'

def search_markers(annotation, start_time=0.0):
    markers = dict()
    markers[AnnotationMarker.PROBE_MARKER] = list()
    markers[AnnotationMarker.OPENED_EYES_MARKER] = list()
    markers[AnnotationMarker.CLOSED_EYES_MARKER] = list()
    for item in annotation:
        if item[-1][0] == AnnotationMarker.OPENED_EYES_MARKER:
            marker = dict()
            marker[TimeMarker.START] = item[0] + start_time
            marker[TimeMarker.END] = item[0] + item[1] + start_time
            markers[AnnotationMarker.OPENED_EYES_MARKER].append(marker)
        if item[-1][0] == AnnotationMarker.CLOSED_EYES_MARKER:
            marker = dict()
            marker[TimeMarker.START] = item[0] + start_time
            marker[TimeMarker.END] = item[0] + item[1] + start_time
            markers[AnnotationMarker.CLOSED_EYES_MARKER].append(marker)
        if item[-1][0] == AnnotationMarker.PROBE_MARKER:
            marker = dict()
            marker[TimeMarker.START] = item[0] + start_time
            marker[TimeMarker.END] = item[0] + item[1] + start_time
            markers[AnnotationMarker.PROBE_MARKER].append(marker)
    return markers


def plot_data(data, time, channels, channel_names=None, scale=None):
    plt.close('all')

    fig, ax = plt.subplots(figsize=(16, 10), nrows=len(channels), ncols=1)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.97, hspace=0.02)

    t = time
    s = data
    for i in channels:
        if scale:
            min_range = 7 * scale * min(s[i]) / (max(s[i]) - min(s[i]))
            max_range = 7 * scale * max(s[i]) / (max(s[i]) - min(s[i]))
        else:
            min_range = min(s[i])
            max_range = max(s[i])
        ax[i - channels[0]].plot(t, s[i])
        ax[i - channels[0]].set_xlim([0, 10])
        ax[i - channels[0]].tick_params(labelsize=7)
        ax[i - channels[0]].set_ylim([min_range, max_range])
        if channel_names:
            ax[i - channels[0]].set_title(channel_names[i], fontsize=7)

    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.05, 0.01, 0.90, 0.02], axisbg=axcolor)
    spos = Slider(axpos, '', min(t), max(t) - 10)

    def update(val):
        pos = spos.val
        for i in channels:
            if scale:
                min_range = 7 * scale * min(s[i]) / (max(s[i]) - min(s[i]))
                max_range = 7 * scale * max(s[i]) / (max(s[i]) - min(s[i]))
            else:
                min_range = min(s[i])
                max_range = max(s[i])
            ax[i - channels[0]].axis([pos, pos + 10, min_range, max_range])
        fig.canvas.draw_idle()

    spos.on_changed(update)
    plt.show()


def correct_time_markers(markers, time_range_to_delete):
    start = time_range_to_delete[0]
    end = time_range_to_delete[1]
    new_markers = dict(markers)
    for annotation_marker in [AnnotationMarker.PROBE_MARKER, AnnotationMarker.OPENED_EYES_MARKER,
                              AnnotationMarker.CLOSED_EYES_MARKER]:
        if len(markers[annotation_marker]) != 0:
            deleted_items = 0
            for item in range(len(markers[annotation_marker])):
                if markers[annotation_marker][item][TimeMarker.START] >= end:
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.START] -= (end - start)
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.END] -= (end - start)
                elif (markers[annotation_marker][item][TimeMarker.START] < start and
                      markers[annotation_marker][item][TimeMarker.END] > end):
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.END] -= (end - start)
                elif (markers[annotation_marker][item][TimeMarker.START] >= start and
                      markers[annotation_marker][item][TimeMarker.START] <= end and
                      markers[annotation_marker][item][TimeMarker.END] > end):
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.START] = start
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.END] -= (end - start)
                elif (markers[annotation_marker][item][TimeMarker.START] < start and
                      markers[annotation_marker][item][TimeMarker.END] >= start and
                      markers[annotation_marker][item][TimeMarker.END] <= end):
                    new_markers[annotation_marker][item - deleted_items][TimeMarker.END] = start
                elif (markers[annotation_marker][item][TimeMarker.START] >= start and
                      markers[annotation_marker][item][TimeMarker.END] <= end):
                    new_markers[annotation_marker] = np.delete(new_markers[annotation_marker], item - deleted_items)
                    deleted_items += 1
    return new_markers


def delete_unused_data(X, time, markers, sample_rate):
    intervals_for_processing = []
    for annotation_marker in [AnnotationMarker.PROBE_MARKER, AnnotationMarker.OPENED_EYES_MARKER]:
        for item in range(len(markers[annotation_marker])):
            intervals_for_processing.append(markers[annotation_marker][item])
    intervals_for_processing.sort()
    for item in range(len(intervals_for_processing) + 1):
        start = 0.0 if (item == 0) else intervals_for_processing[item - 1][TimeMarker.END]
        end = max(time) if (item == len(intervals_for_processing)) else intervals_for_processing[item][TimeMarker.START]
        print("Data from {} to {} seconds will be deleted as useless".format(start, end))
        X = np.delete(X, range(int(sample_rate * start), int(sample_rate * end)), 1)
        time = np.arange(0, len(X[0])) / sample_rate
        markers = correct_time_markers(markers, [start, end])
    return X, time, markers


def print_marker_data(markers):
    ordinal = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
    for item in range(len(markers[AnnotationMarker.OPENED_EYES_MARKER])):
        print("Opened eyes background EEG recording starts at {} sec, ends at {} sec".format(
            round(markers[AnnotationMarker.OPENED_EYES_MARKER][item][TimeMarker.START], 3),
            round(markers[AnnotationMarker.OPENED_EYES_MARKER][item][TimeMarker.END], 3)))
    for item in range(len(markers[AnnotationMarker.CLOSED_EYES_MARKER])):
        print("Closed eyes background EEG recording starts at {} sec, ends at {} sec".format(
            round(markers[AnnotationMarker.CLOSED_EYES_MARKER][item][TimeMarker.START], 3),
            round(markers[AnnotationMarker.CLOSED_EYES_MARKER][item][TimeMarker.END], 3)))
    for item in range(len(markers[AnnotationMarker.PROBE_MARKER])):
        print("{} cognitive probe EEG recording starts at {} sec, ends at {} sec".format(ordinal[item], round(
            markers[AnnotationMarker.PROBE_MARKER][item][TimeMarker.START], 3), round(
            markers[AnnotationMarker.PROBE_MARKER][item][TimeMarker.END], 3)))


def correct_time_sequence(time, signal_sample_rate, signal_annotations):
    for i in range(len(time) - 1):
        delta = time[i + 1] - time[i] - 1 / signal_sample_rate
        for annotation_marker in [AnnotationMarker.PROBE_MARKER, AnnotationMarker.OPENED_EYES_MARKER,
                                  AnnotationMarker.CLOSED_EYES_MARKER]:
            for item in signal_annotations[annotation_marker]:
                if item[TimeMarker.START] >= time[i + 1]:
                    item[TimeMarker.START] -= delta
                if item[TimeMarker.END] >= time[i + 1]:
                    item[TimeMarker.END] -= delta
    return signal_annotations


def import_data(file_names):
    signal_sample_rate = 0.0
    time_data = []
    signal = [[] for i in range(23)]
    signal_annotations = dict()
    signal_annotations[AnnotationMarker.PROBE_MARKER] = list()
    signal_annotations[AnnotationMarker.OPENED_EYES_MARKER] = list()
    signal_annotations[AnnotationMarker.CLOSED_EYES_MARKER] = list()
    start_time = 0.0
    for name in file_names:
        X, sample_rate, sens_lab, time, annotations = edfplus.load_edf(name)
        for i in range(len(signal)):
            signal[i].extend(X[i])
        time_data.extend(time + start_time)
        markers = search_markers(annotations, start_time)
        for annotation_marker in [AnnotationMarker.PROBE_MARKER, AnnotationMarker.OPENED_EYES_MARKER,
                                  AnnotationMarker.CLOSED_EYES_MARKER]:
            signal_annotations[annotation_marker].extend(markers[annotation_marker])
        start_time = time_data[-1] + 1 / sample_rate
        leads = list(sens_lab)
        signal_sample_rate = sample_rate
    corrected_annotations = correct_time_sequence(time_data, signal_sample_rate, signal_annotations)
    corrected_time_data = np.arange(0.0, len(time_data) / signal_sample_rate, 1 / signal_sample_rate)
    return signal, signal_sample_rate, leads, corrected_time_data, corrected_annotations


def load_and_process_data(stimulation):
    # Loading EEG data
    # X, sample_rate, sens_lab, time, annotations = edfplus.load_edf('/Users/alexanderashikhmin/Desktop/EEG data/Чай/0603201703.EDF')
    X, sample_rate, sens_lab, time, annotations = import_data(
        ['/Users/alexanderashikhmin/Desktop/EEG data/' + prefix[stimulation] + '01.EDF',
         '/Users/alexanderashikhmin/Desktop/EEG data/' + prefix[stimulation] + '02.EDF',
         '/Users/alexanderashikhmin/Desktop/EEG data/' + prefix[stimulation] + '03.EDF'])
    # Setting seed to avoid randomization
    np.random.seed(0)

    # Setting the epoch length
    epoch = (int(sample_rate) / 5) * 2

    # Setting overlapping
    overlap = 0.5

    # Plotting EEG data
    # plot_data(X, time, range(17), sens_lab, 10)

    if len(annotations[AnnotationMarker.OPENED_EYES_MARKER]) > 3:
        print("\x1b[6;37;41mWarning!\x1b[0m More that three markers for opened eyes state.")
        print("List of markers: ", markers[AnnotationMarker.OPENED_EYES_MARKER])
    if len(annotations[AnnotationMarker.CLOSED_EYES_MARKER]) > 4:
        print("\x1b[6;37;41mWarning!\x1b[0m More that four markers for closed eyes state.")
        print("List of markers: ", markers[AnnotationMarker.CLOSED_EYES_MARKER])
    if len(annotations[AnnotationMarker.PROBE_MARKER]) > 2:
        print("\x1b[6;37;41mWarning!\x1b[0m More that two markers for cognitive probe.")
        print("List of markers: ", markers[AnnotationMarker.PROBE_MARKER])

    print("\nDeleting unused data...")
    X, time, markers = delete_unused_data(X, time, annotations, sample_rate)
    print("\nMarker data:")
    print_marker_data(markers)
    initial_number_of_epochs = len(X[0]) / epoch

    print("Data parameters:")
    print("Duration of EEG signal is {} seconds".format(len(X[0]) / sample_rate))
    print("Number of epochs is", initial_number_of_epochs)
    print("\nMarker data:")
    print_marker_data(annotations)

    progress = IntProgress(min=0, max=len(X[4]) / epoch, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    print("\nSearching epoch artifacts...")
    current_epoch = 0
    channel_amplitude_range = []
    deviation_from_channel_average = []
    variance = []
    while True:
        if current_epoch % 100 == 0 or current_epoch == len(X[4]) / epoch:
            progress.bar_style = 'success'
            progress.value = current_epoch
            label.value = u'{name} {index} / {size} epochs processed'.format(
                name='Searching epoch artifacts...',
                index=current_epoch,
                size=len(X[4]) / epoch
            )
        if (current_epoch + 1) * epoch <= len(X[4]):
            channel_amplitude_range_epoch = []
            deviation_from_channel_average_epoch = []
            variance_epoch = []
            for i in range(17):
                channel_amplitude_range_epoch.append(
                    np.max(X[i][current_epoch * epoch:(current_epoch + 1) * epoch]) - np.min(
                        X[i][current_epoch * epoch:(current_epoch + 1) * epoch]))
                deviation_from_channel_average_epoch.append(
                    np.mean(X[i][current_epoch * epoch:(current_epoch + 1) * epoch]) - np.mean(X[i]))
                variance_epoch.append(np.std(X[i][current_epoch * epoch:(current_epoch + 1) * epoch]))
            channel_amplitude_range.append(np.mean(channel_amplitude_range_epoch))
            deviation_from_channel_average.append(np.mean(deviation_from_channel_average_epoch))
            variance.append(np.mean(variance_epoch))
        else:
            break
        current_epoch += 1

    channel_amplitude_range_z_scores = zscore(channel_amplitude_range)
    deviation_from_channel_average_z_scores = zscore(deviation_from_channel_average)
    variance_z_scores = zscore(variance)
    channel_amplitude_range_to_delete = [abs(score) > 3.0 for score in channel_amplitude_range_z_scores]
    deviation_from_channel_average_to_delete = [abs(score) > 3.0 for score in deviation_from_channel_average_z_scores]
    variance_to_delete = [abs(score) > 3.0 for score in variance_z_scores]

    to_delete = map(lambda x, y, z: x + y + z, channel_amplitude_range_to_delete,
                    deviation_from_channel_average_to_delete, variance_to_delete)
    count = 0
    for n in range(len(to_delete) - 1, -1, -1):
        if to_delete[n]:
            print("Epoch {} will be deleted due to high Z score".format(n))
            X = np.delete(X, range(n * epoch, (n + 1) * epoch), 1)
            markers = correct_time_markers(markers, [n * epoch / sample_rate, (n + 1) * epoch / sample_rate])
            count += 1
    time = np.arange(0, len(X[0])) / sample_rate
    print("\n{} epochs were deleted, which is equal to {} seconds of signal".format(count, count * epoch / sample_rate))
    print("\nMarker data:")
    print_marker_data(markers)
    # plot_data(X, time, range(17), sens_lab, 10)

    # Making ICA decomposition
    print("\nMaking ICA decomposition...")
    ica = FastICA(n_components=17)
    ica_signal = ica.fit_transform(X[:17].T)

    # Checking if ICA decomposaition is valid
    print("ICA is valid? ", np.allclose(X[:17].T, np.dot(ica_signal, ica.mixing_.T) + ica.mean_))

    # Plotting ICA data and artifact signals

    # plot_data(np.vstack((ica_signal.T, X[19:23])), time, range(21), component_names)

    X_saved = X

    # Removing components highly correlated with artifact signals
    print("\nSearching for artifactual ICA components...")
    component_to_delete = []
    for j in range(19, 23):
        correlation_coefficients = []
        art_signal = X[j]
        for i in range(17):
            comp_signal = ica_signal.T[i]
            correlation_coefficients.append(abs(pearsonr(comp_signal, art_signal)[0]))
        print("\nCorrelations with {} signal: ".format(sens_lab[j]))
        z_scores = zscore(correlation_coefficients)
        for n in range(len(correlation_coefficients)):
            correlation_color_code = ""
            z_score_color_code = ""
            if correlation_coefficients[n] >= 0.1 and correlation_coefficients[n] < 0.3:
                correlation_color_code = "\x1b[6;37;41m"
            elif correlation_coefficients[n] >= 0.3 and correlation_coefficients[n] < 0.5:
                correlation_color_code = "\x1b[6;37;43m"
            elif correlation_coefficients[n] >= 0.5:
                correlation_color_code = "\x1b[6;37;42m"
            if abs(z_scores[n]) >= 2.0 and abs(z_scores[n]) < 3.0:
                z_score_color_code = "\x1b[6;37;41m"
            elif abs(z_scores[n]) >= 3.0:
                z_score_color_code = "\x1b[6;37;42m"
            print(
                "for {} component is " + correlation_color_code + "{}" + "\x1b[0m with Z score " + z_score_color_code + "{}\x1b[0m").format(
                n, correlation_coefficients[n], z_scores[n])
        for n in range(len(z_scores)):
            if abs(z_scores[n]) > 3.0 and correlation_coefficients[n] > 0.2:
                print("Component {} will be zeroed due to high correlation with {} signal".format(n, sens_lab[j]))
                component_to_delete.append(n)

    print("\nPlease, review ICA components, artifactual signals and specify, which component to delete...")
    return [X_saved, ica, ica_signal, sample_rate, sens_lab, time, markers, count, initial_number_of_epochs]


def cleaning_data(ica_composition, input_ica_signal, input_sample_rate, input_markers, component_to_delete, count, initial_number_of_epochs):
    epoch = (int(input_sample_rate)/5)*2
    markers = input_markers
    for n in component_to_delete:
        print "Zeroing {} component of ICA decomposition".format(n)
        input_ica_signal.T[n] = 0

    #Recovering EEG signals from ICA components
    print "\nRecovering signals from ICA decomposition..."
    cleaned_data = ica_composition.inverse_transform(input_ica_signal)
    X = cleaned_data.T

    progress_channel = IntProgress(min=0, max=17, value=0)
    label_channel = HTML()
    box_channel = VBox(children=[label_channel, progress_channel])
    display(box_channel)

    print "\nSearching single epoch artifacts in single channel..."
    epoch_to_delete = []
    for i in range(17):
        progress_channel.bar_style = 'success'
        progress_channel.value = i + 1
        label_channel.value = u'{name} {index} / {size} channels processed'.format(
                            name='Searching single epoch artifacts in single channel... ',
                            index=i + 1,
                            size=17
                        )
        channel_amplitude_range = []
        deviation_from_channel_average = []
        variance = []
        median_slope = []
        current_epoch = 0
        while True:
            if (current_epoch + 1)*epoch <= len(X[i]):
                channel_amplitude_range.append(np.max(X[i][current_epoch*epoch:(current_epoch + 1)*epoch]) - np.min(X[i][current_epoch*epoch:(current_epoch + 1)*epoch]))
                deviation_from_channel_average.append(np.mean(X[i][current_epoch*epoch:(current_epoch + 1)*epoch]) - np.mean(X[i]))
                variance.append(np.std(X[i][current_epoch*epoch:(current_epoch + 1)*epoch]))
                median_slope.append(np.median(np.diff(X[i][current_epoch*epoch:(current_epoch + 1)*epoch])))
            else:
                break
            current_epoch += 1

        channel_amplitude_range_z_scores = zscore(channel_amplitude_range)
        deviation_from_channel_average_z_scores = zscore(deviation_from_channel_average)
        variance_z_scores = zscore(variance)
        median_slope_z_scores = zscore(median_slope)

        channel_amplitude_range_to_delete = [abs(score) > 3.0 for score in channel_amplitude_range_z_scores]
        deviation_from_channel_average_to_delete = [abs(score) > 3.0 for score in deviation_from_channel_average_z_scores]
        variance_to_delete = [abs(score) > 3.0 for score in variance_z_scores]
        median_slope_to_delete = [abs(score) > 3.0 for score in median_slope_z_scores]

        to_delete = map(lambda x, y, z, t: x + y + z + t, channel_amplitude_range_to_delete, deviation_from_channel_average_to_delete, variance_to_delete, median_slope_to_delete)
        epoch_to_delete.append(to_delete)

    new_count = 0
    for i in range(len(epoch_to_delete[0]) - 1, -1, -1):
        arr = [epoch_to_delete[j][i] for j in range(len(epoch_to_delete))]
        if any(arr):
            print "Epoch {} will be deleted due to high Z score".format(i)
            X = np.delete(X, range(i*epoch,(i + 1)*epoch), 1)
            markers = correct_time_markers(input_markers, [i*epoch/input_sample_rate, (i + 1)*epoch/input_sample_rate])
            new_count += 1
    print "\n{} epochs were deleted, which is equal to {} seconds of signal".format(new_count, new_count*epoch/input_sample_rate)
    time = np.arange(0, len(X[0]))/input_sample_rate
    print "\nMarker data:"
    print_marker_data(input_markers)
    #plot_data(X, time, range(17), sens_lab, 10)

    print "\nEEG signal cleaning summary:"
    print "Total number of deleted epochs is ", new_count + count
    print "Total duration of deleted signal is {} seconds".format((new_count + count)*epoch/input_sample_rate)
    print "{}% of epochs were deleted".format(round((new_count + count)/float(initial_number_of_epochs)*100, 1))
    print "{} components of ICA decomposition were zeroed".format(len(component_to_delete))
    print "\nMarker data:"
    print_marker_data(markers)
    return X, time, markers
