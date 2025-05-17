from IPython.display import HTML, display


def inject_jupyter_style():
    """
    make font jupyter widgets (like tqdm) same as vscode theme
    """

    display(
        HTML(
            """
    <style>
    .cell-output-ipywidget-background {
        background-color: transparent !important;
    }
    :root {
        --jp-widgets-color: var(--vscode-editor-foreground);
        --jp-widgets-font-size: var(--vscode-editor-font-size);
    }
    </style>
    """
        )
    )
