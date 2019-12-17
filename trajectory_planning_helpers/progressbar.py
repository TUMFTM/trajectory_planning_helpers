import sys


def progressbar(i: int,
                i_total: int,
                prefix: str = '',
                suffix: str = '',
                decimals: int = 1,
                length: int = 50) -> None:
    """
    author:
    Tim Stahl

    .. description::
    Commandline progressbar (to be called in a for loop).

    .. inputs::
    :param i:           current iteration / progress index.
    :type i:            int
    :param i_total:     maximum iteration number / progress (where 100% should be reached).
    :type i_total:      int
    :param prefix:      prefix string to be displayed right in front of progressbar.
    :type prefix:       str
    :param suffix:      suffix string to be displayed behind the progressbar.
    :type suffix:       str
    :param decimals:    number of decimals behind comma (of printed percentage).
    :type decimals:     int
    :param length:      length of progressbar (in character spaces).
    :type length:       int
    """

    # Calculate current percentage based on i and i_total
    percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(i_total)))

    # elements to be filled based on length and current status
    filled_length = int(length * i // i_total)

    # generate progress sting
    bar = '█' * filled_length + '-' * (length - filled_length)

    # print ("\r" for same line)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))

    # new line when done
    if i >= i_total:
        print()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
