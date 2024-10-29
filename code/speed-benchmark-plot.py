import pathlib
import subprocess
import datetime

from matplotlib import pyplot as plt
import labellines
import polars as pl
import numpy as np

import textbox

# config
single_figure = True

# reset matplotlib
plt.close('all')
plt.rcdefaults()

# cycler for plots
cycler = plt.cycler(
    color=['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:5], # from style tableau-colorblind10
    # color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:5],
    # color=5 * ['#c00'],
    linestyle=['-', '--', '-.', ':', '-'],
    marker=5 * ['.'],
    markerfacecolor=5 * ['none'],
)

# data printed by:
# - speed-benchmark-bartz.py
# - speed-benchmark-bartz.ipynb
# - speed-benchmark-dbarts.py
# - speed-benchmark-xgboost.py
# - speed-benchmark-xgboost.ipynb
results = [

    #### bartz cpu ####
    {
        'package': 'bartz',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'n/p': 10,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'time_per_iter': [2.1762499818578364e-05, 2.1206249948590993e-05, 2.4572950133006087e-05, 2.2977099797572008e-05, 2.59666998317698e-05, 3.849579989037011e-05, 6.136875017546118e-05, 0.00018866040009015705, 0.0004937416499160463, 0.0012653791498451028, 0.0038568853997276165, 0.013066824999987148, 0.04996391454988043, 0.19541140830006043, 0.7620472354497906, 3.3000546854000277, 13.881264585399913]},
    },
    {
        'package': 'bartz',
        'device_kind': 'cpu',
        'ntree': 200,
        'p': 100,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432], 'time_per_iter': [0.000727616665729632, 0.0007028388674370945, 0.0007180555335556467, 0.0007281333324499428, 0.0007646861330916484, 0.00083151946698005, 0.0009492360676328341, 0.001074274998002996, 0.0013880860642530024, 0.0019346138656449814, 0.0031320055364631116, 0.005423561134375632, 0.010226794468083729, 0.01993252773148318, 0.038615152732624364, 0.08457702780142426, 0.1624668249976821, 0.30419243613335617, 0.5948999361329091, 1.145889908335327, 2.3735731972653107, 4.5692836722009815, 9.164452386069266, 19.06538725553158, 40.457980688863124]},
    },
    {
        'package': 'bartz',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'p': 100,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], 'time_per_iter': [2.1141666608552137e-05, 2.065833347539107e-05, 2.0658329594880344e-05, 2.2033334244042634e-05, 3.095833429445823e-05, 3.970553322384755e-05, 6.270280185466011e-05, 0.0001943360664881766, 0.00047719446398938696, 0.0012842055992223322, 0.003958683332893997, 0.013240697199944407, 0.050717580601728214, 0.1993615527986549, 0.7803820916684344, 3.409753433331692, 13.167540991667192, 53.091798072199644]},
    },
    {
        'package': 'bartz',
        'device_kind': 'cpu',
        'ntree': 200,
        'n/p': 10,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], 'time_per_iter': [0.0007245999998607052, 0.000741347200043189, 0.000743516666504244, 0.0007318138668779284, 0.0007792943996416094, 0.0008170916664918574, 0.000957727799929368, 0.0010966249998697701, 0.0014009388668152192, 0.001969561066653114, 0.0032112194002062704, 0.005387425000177851, 0.010319855533210406, 0.019731672199850437, 0.03852081106645831, 0.08450916386645986, 0.16158210560000347, 0.32646868613325447]},
    },

    #### bartz gpu ####
    {
        'package': 'bartz',
        'device_kind': 'NVIDIA A100-SXM4-40GB',
        'n/ntree': 8,
        'n/p': 10,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], 'time_per_iter': [0.0017378434000003531, 0.001968255933331875, 0.0021415648666675224, 0.0019234018666641835, 0.0019933754666681125, 0.0020198342666617464, 0.0023776924666663035, 0.002966313000001719, 0.0036372135333370653, 0.004851416400000138, 0.008151953933338518, 0.011411968400003995, 0.021465109533331393, 0.043960155599999474, 0.09103459800000262, 0.17994991893332704, 0.39830770986666264, 1.2350581426666698]},
    },
    {
        'package': 'bartz',
        'device_kind': 'NVIDIA A100-SXM4-40GB',
        'ntree': 200,
        'n/p': 10,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'time_per_iter': [0.0061308731333307755, 0.006286818000004738, 0.0063109865333368965, 0.005933725399995637, 0.006309372266665984, 0.006630090066672286, 0.00637683706666697, 0.006374272933332274, 0.006197058066663885, 0.006509123599994382, 0.006682615399995484, 0.00706864173333391, 0.006679336199999853, 0.007151935533336958, 0.006672989266667173, 0.006699655333333491, 0.008960054799998337]},
    },
    {
        'package': 'bartz',
        'device_kind': 'NVIDIA A100-SXM4-40GB',
        'ntree': 200,
        'p': 100,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864], 'time_per_iter': [0.005969145200000033, 0.005576269666668547, 0.006125143200006278, 0.006028980866669069, 0.005902854999999363, 0.005827513466662519, 0.006359983466662319, 0.006068931333334149, 0.0065649674000042065, 0.006004334400002639, 0.00681978986666157, 0.006379730000003292, 0.006348076933333383, 0.006986279466665716, 0.006748121666669249, 0.007119519666669779, 0.007778303066667529, 0.009856747266667298, 0.009997110999999373, 0.012346643266672194, 0.01687075579999752, 0.02275558019999456, 0.04617121433333674, 0.08688909446666457, 0.16264456686666715, 0.3206280234666641]},
    },
    {
        'package': 'bartz',
        'device_kind': 'NVIDIA A100-SXM4-40GB',
        'n/ntree': 8,
        'p': 100,
        'maxdepth': 6,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], 'time_per_iter': [0.0016933792000012697, 0.002215988133332303, 0.0018291676666668385, 0.001962961466665547, 0.0021024648000017502, 0.0021090906666662097, 0.002308721733334096, 0.0027841375333328717, 0.0035117906666660777, 0.004817838600001778, 0.0076736030000006394, 0.012028651133330943, 0.021966619533331292, 0.04362189186666683, 0.0912944418000014, 0.18079169906666645, 0.4066115397333306, 1.1911996044666655]},
    },

    #### dbarts cpu ####
    {
        'package': 'dbarts',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], 'time_per_iter': [0.00023450625012628735, 2.855209750123322e-05, 2.8031249530613423e-05, 2.959789999295026e-05, 3.537290031090379e-05, 3.844374732580036e-05, 5.957295070402324e-05, 8.93270509550348e-05, 0.00019312709919176995, 0.0005519312515389174, 0.0019083458493696526, 0.007778072901419364, 0.031546308298129586, 0.19093005624890794, 0.8838882625015685, 6.3748561416985465], 'time_init': [0.021258249995298684, 0.003336500027216971, 0.0033135419944301248, 0.003519332967698574, 0.003913249995093793, 0.004685749998316169, 0.006171083019580692, 0.008969040994998068, 0.01607695803977549, 0.026519040984567255, 0.051113000023178756, 0.10209316702093929, 0.19770775001961738, 0.4027342919725925, 0.8524772920063697, 1.987792707979679]},
    },
    {
        'package': 'dbarts',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], 'time_per_iter': [1.8666699179448188e-05, 1.835415023379028e-05, 1.8637499306350946e-05, 1.984584960155189e-05, 2.0864600082859396e-05, 2.6835399330593647e-05, 4.777919966727495e-05, 8.571250073146075e-05, 0.00019837295112665743, 0.0005999396002152934, 0.0023688625020440667, 0.008729985399986618, 0.03217564795049839, 0.19840167704969644, 0.8714290999982041], 'time_init': [0.0033769579604268074, 0.003345875011291355, 0.003198999969754368, 0.002934916003141552, 0.0029537910013459623, 0.0029653329984284937, 0.0033324999967589974, 0.004980500030796975, 0.009636000031605363, 0.029525000019930303, 0.10614124999847263, 0.40960387501399964, 1.6188469170010649, 6.399999040993862, 25.917172125016805]},
    },
    {
        'package': 'dbarts',
        'device_kind': 'cpu',
        'ntree': 200,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], 'time_per_iter': [0.0001462916494347155, 0.00017224789771717042, 0.0001738979510264471, 0.00018752914911601692, 0.00023032079916447402, 0.00024178330204449594, 0.0002948604000266641, 0.0003873458510497585, 0.000536431249929592, 0.000920635400689207, 0.0020700145978480577, 0.0029997708508744834, 0.005922333349008113, 0.020681637499365024, 0.042465431249002, 0.09021354790020268], 'time_init': [0.002977791999001056, 0.0030144170159474015, 0.0029204580350778997, 0.0030251670395955443, 0.003184291999787092, 0.003033250046428293, 0.0033897499670274556, 0.00468887499300763, 0.009154166036751121, 0.02722075005294755, 0.10315391601761803, 0.39423791703302413, 1.5584344160160981, 6.299652374989819, 25.134852332994342, 109.59230604197364]},
    },
    {
        'package': 'dbarts',
        'device_kind': 'cpu',
        'ntree': 200,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152], 'time_per_iter': [0.0001762812491506338, 0.00021681459911633284, 0.0002163791476050392, 0.0002304021007148549, 0.00024322704994119703, 0.000294706248678267, 0.00032504789996892216, 0.000414775000535883, 0.0005606770515441895, 0.0009014479001052678, 0.0015013124997494743, 0.0029643771005794404, 0.006449650001013651, 0.01847458124975674, 0.04209256664908025, 0.08552272499946412, 0.17579324374964928, 0.3557762937474763, 0.7114472770510474, 1.5499466979003045, 5.885803481249605], 'time_init': [0.0036515420069918036, 0.003399292007088661, 0.0034064999781548977, 0.004066416004206985, 0.003926707955542952, 0.004591084027197212, 0.006131625035777688, 0.009822332998737693, 0.016204541956540197, 0.02733266697032377, 0.05080783402081579, 0.09800862503470853, 0.19332329201279208, 0.38770862499950454, 0.757022165984381, 1.5796924169990234, 3.1383147909655236, 6.485036250029225, 12.48952891601948, 25.157118958013598, 52.20104849996278]},
    },

    #### xgboost cpu ####
    {
        'package': 'xgboost',
        'device_kind': 'cpu',
        'ntree': 200,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608], 'time_per_iter': [0.00876412505749613, 0.007772832992486656, 0.010663166991434991, 0.01665741699980572, 0.030680415977258235, 0.0612264170194976, 0.14593104098457843, 0.4392255420098081, 0.787650250014849, 1.3620359999476932, 2.405932000023313, 2.8868795830057934, 3.382152167032473, 4.06668616598472, 5.113125542004127, 7.079545624961611, 10.67216458299663, 17.77914687496377, 32.209680624946486, 60.43522858299548, 116.20660337503068, 230.3559957499965, 480.84084704099223]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'cpu',
        'ntree': 200,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], 'time_per_iter': [0.0063722910126671195, 0.008814833010546863, 0.008900584012735635, 0.01074058300582692, 0.00965249998262152, 0.011501999979373068, 0.027858000015839934, 0.1291981670074165, 0.42551712499698624, 1.4037330410210416, 4.478746125008911, 10.841826875053812, 24.761165499978233, 60.71221075003268, 149.84925383300288, 429.61977312498493]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'time_per_iter': [0.0017395419999957085, 0.0016223330167122185, 0.0018352920305915177, 0.0024963750038295984, 0.0053237079991959035, 0.013163708033971488, 0.05189345800317824, 0.265837708953768, 0.5657192500075325, 1.2687034160480835, 2.326980084006209, 4.24440304201562, 8.311429959023371, 17.287880166026298, 40.512791124987416, 111.52969937497983, 355.5012948749936]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'cpu',
        'n/ntree': 8,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], 'time_per_iter': [0.0017285420326516032, 0.0016570829902775586, 0.0028143340023234487, 0.002002333989366889, 0.002767833007965237, 0.0038621669518761337, 0.010556208027992398, 0.07200091698905453, 0.3160133329802193, 1.2948585409903899, 4.425855082983617, 15.669485124992207, 59.212344707979355, 261.51704374997644, 1195.9874566670042]},
    },

    #### xgboost gpu ####
    {
        'package': 'xgboost',
        'device_kind': 'NVIDIA L4',
        'ntree': 200,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432], 'time_per_iter': [1.8384869089999825, 0.09282492699998102, 0.09152431000001116, 0.1184428440000147, 0.10354921200001854, 0.10927933899995423, 0.12866700500001116, 0.16489932799998996, 0.2560162739999896, 0.35124912399999175, 0.5180471279999779, 0.5706175680000456, 0.5939843860000451, 0.6497193529999663, 0.7636446430000206, 0.9073905339999442, 1.1463312289999976, 1.7551562740000008, 3.102892827000005, 5.106467790000011, 9.882363772000076, 19.773469773999977, 38.77552712600004, 79.15513312500002, 158.77939497199998]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'NVIDIA L4',
        'ntree': 200,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], 'time_per_iter': [0.0956978839999465, 0.09591787299996213, 0.11113751000016237, 0.10431300899995222, 0.10716749600010189, 0.11058036600002197, 0.11372449999998935, 0.14877256599993416, 0.20156954899994162, 0.3708372999999483, 0.7219977830000062, 1.3196680020000713, 2.827851045999978, 6.197613348000004, 13.526395346000072, 35.13332850100005, 103.03530292300002]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'NVIDIA L4',
        'n/ntree': 8,
        'p': 100,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288], 'time_per_iter': [0.017225239000026704, 0.012436800000159565, 0.011068175999980667, 0.01141922900001191, 0.014379351000116003, 0.019825130000072022, 0.03360716999986835, 0.07364502400014317, 0.14924642100004348, 0.30206698799997866, 0.5931230809999306, 0.9666950250000355, 1.604646867000156, 3.0671376179998333, 6.896505289000061, 14.582030843999974, 30.929300067999975, 83.85839221699985, 286.068365888]},
    },
    {
        'package': 'xgboost',
        'device_kind': 'NVIDIA L4',
        'n/ntree': 8,
        'n/p': 10,
        'results': {'n': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], 'time_per_iter': [0.013073125000119035, 0.0073670649999257876, 0.007765260999804013, 0.01017437800010157, 0.01251614199964024, 0.01849064999987604, 0.030551524999737012, 0.055883256000015535, 0.12073724599986235, 0.3046055640002123, 0.7697822579998501, 2.0684659249996002, 7.4423875380002755, 27.64662913600023, 96.12471483799982, 439.9091482900003]},
    },
]

# merge data into one long-format dataframe
tables = []
for things in results:
    tables.append(
        pl.DataFrame(things['results'])
        .with_columns([
            pl.lit(v).alias(k)
            for k, v in things.items()
            if k != 'results'
        ])
    )

df = (
    pl.concat(tables, how='diagonal')
    .with_columns(pl.col('device_kind').replace({
        'NVIDIA L4': 'L4',
        'NVIDIA A100-SXM4-40GB': 'A100',
    }))
    .with_columns(case=pl.concat_str('package', 'device_kind', separator='-'))
)

keynames = ['n/ntree', 'ntree', 'n/p', 'p']
groups = list(df.group_by(keynames, maintain_order=True))

if single_figure:
    fig, axs = plt.subplots(2, 2,
        figsize=[8.5, 8.5],
        num=f'speed-benchmark-plot',
        clear=True,
        layout='constrained',
        sharex=True,
        sharey=True,
    )
    axs = axs.flat
    figs = [fig]
else:
    axs = []
    figs = []
    for i in range(len(groups)):
        fig, ax = plt.subplots(
            figsize=[5, 4],
            num=f'benchmark-plot-{i}',
            clear=True,
            layout='constrained',
        )
        axs.append(ax)
        figs.append(fig)
axs[1], axs[3] = axs[3], axs[1]

if single_figure:
    ax = axs[0]
    ax.set(xscale='log', yscale='log')
    ax.set_xlim(10, 2 * 10 ** 8)
    ax.set_ylim(10 ** -5, 10 ** 3.5)

for ax, (keys, group) in zip(axs, groups):
    ax.set_prop_cycle(cycler)
    for (case,), data in group.sort('case').group_by('case', maintain_order=True):
        ax.plot(data['n'], data['time_per_iter'], label=case)
    
    ss = ax.get_subplotspec()
    
    if ss.is_last_row():
        ax.set_xlabel('n')
    if ss.is_first_col():
        ax.set_ylabel('Time per iteration [s]')
    
    if not single_figure:
        ax.set(xscale='log', yscale='log')
        ax.set_xlim(10, ax.get_xlim()[1])
        xvals = None
    elif ss.is_first_row() and ss.is_first_col():
        xvals = [200, 3500, 300, 19_000, 5000]
    elif ss.is_first_row() and ss.is_last_col():
        xvals = [200, 100, 300, 40_000, 10_000]
    elif ss.is_last_row() and ss.is_first_col():
        xvals = [200, 3500, 300, 19_000, 5000]
    else:
        xvals = [200, 100, 300, 40_000, 40_000]
    labellines.labelLines(ax.get_lines(), xvals=xvals, outline_width=3)
    
    ax.grid(linestyle='--')
    ax.grid(which='minor', linestyle=':')

    textbox.textbox(ax, '\n'.join(
        f'{name}={value}'
        for name, value in zip(keynames, keys)
        if value is not None
    ), loc='upper left')

for fig in figs:
    fig.show()

# save figures
script = pathlib.Path(__file__)
outdir = script.with_suffix('')
outdir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()[:7]
for fig in figs:
    figname = f'{commit}_{timestamp}_{fig.get_label()}.pdf'
    fig.savefig(outdir / figname)