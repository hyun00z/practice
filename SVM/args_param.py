from gooey import GooeyParser

def add(parser: GooeyParser = GooeyParser()) -> GooeyParser:
    parser.add_argument(
        "--kernel",
        metavar='to control the amount of shrinkage',
        help='The larger the value of alpha, the greater the amount of shrinkage',
        choices=['linear','poly','rbf']
    )

    parser.add_argument(
        "--C", type=float,
        metavar='Regularization parameter',
        default=1.0,
        help='The strength of the regularization is inversely proportional to C.',
        gooey_options={
            'validator': {
                'test': 'float(user_input) > 0',
                'message': 'Must be positive number.'
            }
        }
    )

    parser.add_argument(
        "--degree", type=int,
        metavar='Degree',
        default=3,
        help='Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.',
        gooey_options={
            'validator': {
                'test': 'int(user_input) > 0',
                'message': 'Must be positive number.'
            }
        }
    )

    return parser
