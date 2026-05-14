import traceback


def click_success(result):
    """
    Helper function to verify that a Click command executed successfully (exit code 0).

    If the command failed, this function prints the output and traceback to aid in debugging
    before returning False.
    """
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0
