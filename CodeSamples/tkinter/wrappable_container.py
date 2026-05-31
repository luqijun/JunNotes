"""
功能需求：
python使用tkinter库，整体界面包含主体内容区域和底部区域。
在主体内容区域，使用一个画布且带有水平垂直滚动条的示例，画布中每条内容横向自动填充画布的剩余空间，垂直滚动条和水平滚动条自动显示和隐藏。
在底部区域，以固定高度，包含Cancel和OK按钮， 点击Cancel按钮或窗口右上角的关闭按钮关闭窗口并返回，
点击OK按钮时，只有满足一定条件才能关闭，否则弹窗提示不满足条件，且不关闭窗口。窗口关闭后要有对应的返回值。

(主体内容区域改为使用一个类似于wrappannel的控件，使得内容可以根据容器宽度自适应显示。)

"""
import tkinter as tk
from tkinter import messagebox, ttk


class ScrollableApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter Scrollable Canvas Example")
        self.root.geometry("800x600")

        # 用于存储关闭时的返回值
        self.return_value = None

        # 创建主容器
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 创建主体内容区域
        self.create_content_area()

        # 创建底部区域
        self.create_bottom_area()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 添加一些示例内容
        self.add_sample_content()

    def create_content_area(self):
        """创建带有滚动条的内容区域"""
        # 创建内容区域的Frame
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建画布
        self.canvas = tk.Canvas(self.content_frame, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建垂直滚动条
        self.v_scrollbar = ttk.Scrollbar(
            self.content_frame,
            orient=tk.VERTICAL,
            command=self.canvas.yview
        )
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建水平滚动条
        self.h_scrollbar = ttk.Scrollbar(
            self.main_container,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview
        )
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

        # 配置画布的滚动
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )

        # 创建内部Frame用于放置内容
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.inner_frame,
            anchor=tk.NW,
            width=self.canvas.winfo_reqwidth()  # 初始宽度
        )

        # 绑定事件以更新滚动区域和自动显示/隐藏滚动条
        self.inner_frame.bind('<Configure>', self.update_scrollregion)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self.on_canvas_mousewheel)

    def create_bottom_area(self):
        """创建底部区域"""
        self.bottom_frame = ttk.Frame(self.main_container, height=60)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.bottom_frame.pack_propagate(False)  # 固定高度

        # 创建按钮
        ttk.Button(
            self.bottom_frame,
            text="Cancel",
            command=self.on_cancel
        ).pack(side=tk.RIGHT, padx=5, pady=10)

        ttk.Button(
            self.bottom_frame,
            text="OK",
            command=self.on_ok
        ).pack(side=tk.RIGHT, padx=5, pady=10)

        # 添加示例条件输入框
        self.condition_var = tk.StringVar()
        condition_frame = ttk.Frame(self.bottom_frame)
        condition_frame.pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Label(condition_frame, text="输入'OK'才能关闭:").pack(side=tk.LEFT)
        ttk.Entry(
            condition_frame,
            textvariable=self.condition_var,
            width=20
        ).pack(side=tk.LEFT, padx=5)

    def add_sample_content(self):
        """添加示例内容到画布"""
        colors = ['#f0f0f0', '#e0e0e0', '#d0d0d0', '#c0c0c0']

        for i in range(20):
            # 每条内容使用Frame容器
            item_frame = ttk.Frame(self.inner_frame, relief=tk.RAISED, borderwidth=1)
            item_frame.pack(fill=tk.X, padx=5, pady=5)

            # 让Frame的背景色交替
            # if i % 2 == 0:
            #     style_name = f"ItemStyle{i}"
            #     style = ttk.Style()
            #     style.configure(style_name, background=colors[i % len(colors)])
            #     item_frame.configure(style=style_name)

            # 添加内容
            ttk.Label(
                item_frame,
                text=f"项目 {i+1}: 这是示例内容，会自动填充横向空间",
                font=('Arial', 10)
            ).pack(side=tk.LEFT, padx=10, pady=10)

            # 添加一些控件
            ttk.Button(
                item_frame,
                text=f"按钮 {i+1}",
                command=lambda idx=i: self.on_item_button_click(idx)
            ).pack(side=tk.RIGHT, padx=10, pady=5)

            ttk.Checkbutton(
                item_frame,
                text=f"选择项 {i+1}"
            ).pack(side=tk.RIGHT, padx=10, pady=5)

    def on_item_button_click(self, index):
        """示例按钮点击事件"""
        print(f"项目 {index+1} 的按钮被点击")

    def update_scrollregion(self, event=None):
        """更新滚动区域并自动显示/隐藏滚动条"""
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # 自动显示/隐藏滚动条
        self.auto_show_scrollbars()

    def on_canvas_configure(self, event):
        """当画布大小改变时调整内部Frame的宽度"""
        # 获取画布的新宽度
        canvas_width = event.width

        # 更新内部Frame的宽度以匹配画布宽度
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

        # 重新计算滚动区域
        self.update_scrollregion()

    def on_canvas_mousewheel(self, event):
        """鼠标滚轮滚动事件"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def auto_show_scrollbars(self):
        """根据内容自动显示或隐藏滚动条"""
        # 获取内容区域和画布的尺寸
        content_bbox = self.canvas.bbox("all")

        if content_bbox:
            content_width = content_bbox[2] - content_bbox[0]
            content_height = content_bbox[3] - content_bbox[1]

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # 显示/隐藏垂直滚动条
            if content_height > canvas_height:
                self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.v_scrollbar.pack_forget()

            # 显示/隐藏水平滚动条
            if content_width > canvas_width:
                self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
            else:
                self.h_scrollbar.pack_forget()

    def check_condition(self):
        """检查关闭条件"""
        # 示例条件：输入框必须包含"OK"
        return self.condition_var.get().strip().upper() == "OK"

    def on_ok(self):
        """OK按钮点击事件"""
        if self.check_condition():
            self.return_value = "ok"
            self.root.destroy()
        else:
            messagebox.showwarning(
                "条件不满足",
                "请在输入框中输入'OK'才能关闭窗口！"
            )

    def on_cancel(self):
        """Cancel按钮点击事件"""
        self.return_value = "cancel"
        self.root.destroy()

    def on_closing(self):
        """窗口关闭事件"""
        self.return_value = "cancel"
        self.root.destroy()

    def run(self):
        """运行应用并返回结果"""
        self.root.mainloop()
        return self.return_value


# 创建主窗口并运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = ScrollableApp(root)

    # 运行应用并获取返回值
    result = app.run()

    # 打印返回值
    print(f"窗口关闭，返回值: {result}")
