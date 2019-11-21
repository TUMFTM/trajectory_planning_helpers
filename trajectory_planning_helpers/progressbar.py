import sys


def progressbar(i: int,
                i_total: int,
                prefix: str = '',
                suffix: str = '',
                decimals: int = 1,
                length: int = 50) -> None:
    """
    Author:
    Tim Stahl

    Description:
    Commandline progressbar (to be called in a for loop).

    Input:
    i:                    current iteration / progress index.
    i_total:              maximum iteration number / progress (where 100% should be reached).
    prefix:               prefix string to be displayed right in front of progressbar.
    suffix:               suffix string to be displayed behind the progressbar.
    decimals:             number of decimals behind comma (of printed percentage).
    length:               length of progressbar (in character spaces).
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
