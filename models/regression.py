def calc_hlfpa_effort(file, interface, inp, out, enq):
    hlfpa = 7 * file + 5 * interface + 4 * inp + 5 * out + 4 * enq
    return 74.9 * (hlfpa ** 0.58)

def calc_ifpa_effort(file, interface):
    ifpa = 35 * file + 15 * interface
    return 539.99 * (ifpa ** 0.272)

def calc_mfp_effort(file, interface, inp, out, enq):
    fp = file + interface + inp + out + enq
    return 87.42 * (fp ** 0.56)
