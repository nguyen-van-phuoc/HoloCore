from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=8):
        super().__init__(parent)

        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

        self.items = []

    def addItem(self, item: QLayoutItem):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize(0, 0)

        for item in self.items:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()

        size += QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom()
        )

        return size

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()

        x = rect.x() + margins.left()
        y = rect.y() + margins.top()

        line_height = 0

        spacing = self.spacing()

        effective_width = (
            rect.width()
            - margins.left()
            - margins.right()
        )

        for item in self.items:
            widget = item.widget()

            item_size = item.sizeHint()

            next_x = x + item_size.width()

            if (
                line_height > 0
                and next_x > rect.x() + margins.left() + effective_width
            ):
                x = rect.x() + margins.left()
                y += line_height + spacing

                next_x = x + item_size.width()
                line_height = 0

            if not test_only:
                item.setGeometry(
                    QRect(
                        x,
                        y,
                        item_size.width(),
                        item_size.height()
                    )
                )

            x = next_x + spacing
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + margins.bottom()