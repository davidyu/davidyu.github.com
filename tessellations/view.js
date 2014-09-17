/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="./model.ts" />

// quick-n-dirty
var Vec2 = (function () {
    function Vec2(x, y) {
        this.x = x;
        this.y = y;
    }
    Vec2.prototype.toString = function () {
        return this.x + "," + this.y;
    };
    return Vec2;
})();

var View;
(function (View) {
    var Colorizer = (function () {
        function Colorizer() {
            this.scale = {};
            this.scale[1 /* EMPTY */] = chroma.scale(['#D7FAF3', '#F3F4E5', '#FFFFFF']);
            this.scale[2 /* REGULAR */] = chroma.scale(['#4BF920', '#1DE5A2', '#48CC20', '#18BC49', '#0DAD6D']);
            this.scale[5 /* LAVA */] = chroma.scale(['#AE5750', '#F96541', '#FF7939']);
            this.scale[4 /* DEACTIVATED */] = chroma.scale(['#64585A', '#64585A']);
        }
        Colorizer.prototype.fromTile = function (t) {
            return this.scale[t.type](t.value / 9).hex();
        };
        return Colorizer;
    })();

    function fromModel(grid) {
        if (grid instanceof Model.Square)
            return new SquareView(grid);
        else if (grid instanceof Model.Hex)
            return new HexView(grid);
    }
    View.fromModel = fromModel;

    var SquareView = (function () {
        function SquareView(square) {
            this.model = square;
        }
        SquareView.prototype.draw = function (canvas) {
            // TODO implement me
        };
        SquareView.prototype.getDOMElements = function () {
            // TODO implement me
            return null;
        };
        SquareView.prototype.getDOMElement = function (x, y) {
            // TODO implement me
            return null;
        };
        return SquareView;
    })();
    View.SquareView = SquareView;

    var HexView = (function () {
        function HexView(hex) {
            this.model = hex;
            this.colorizer = new Colorizer();
        }
        HexView.prototype.toAbsX = function (q) {
            return q + this.model.gridr;
        };

        HexView.prototype.toAbsY = function (r) {
            return r + this.model.gridr;
        };

        HexView.prototype.drawTile = function (canvas, q, r, e) {
            var fudgeMargin = 10;
            var size = (canvas.attr("width") - fudgeMargin) / this.model.diameter();

            // size is the diameter of the inscribed circle. radius refers to the radius of the
            // circumscribed circle
            var radius = size / Math.sin(60 * Math.PI / 180) / 2;

            // distance between adjacent (heightwise) hex cells
            var yd = radius * Math.cos(60 * Math.PI / 180);

            // distance between adjacent (withwise) hex cells
            var xd = radius * Math.sin(60 * Math.PI / 180);

            var yOffset = radius;
            var xOffset = xd;

            var x = this.toAbsX(q);
            var y = this.toAbsY(r);

            var center_x = size * x + xd * r + xOffset;
            var center_y = (yd + radius) * y + yOffset;

            var cell = canvas.group().transform({ x: center_x, y: center_y });

            cell.data = { x: x, y: y, q: q, r: r };

            var pts = [new Vec2(0, -radius), new Vec2(xd, -yd), new Vec2(xd, yd), new Vec2(0, radius), new Vec2(-xd, yd), new Vec2(-xd, -yd)];
            var ptstr = pts.reduce(function (p1, p2, i, v) {
                return p1.toString() + " " + p2.toString();
            }, "");

            var hex = cell.polygon(ptstr);

            hex.attr({
                'fill': this.colorizer.fromTile(e),
                'stroke': '#fff',
                'stroke-width': 2 });

            var text = cell.plain(e.type != 1 /* EMPTY */ ? e.value.toString() : "").fill({ color: '#fff' }).transform({ x: -3.5, y: 3.5 });

            return cell;
        };

        HexView.prototype.draw = function (canvas) {
            if (this.model == null)
                return;

            this.cells = [];

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(q, r);
                    if (e.type == 0 /* OUT_OF_BOUNDS */) {
                        this.cells[this.model.toFlat(q, r)] = null; // standin
                    }
                }
            }

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(q, r);
                    if (e.type == 1 /* EMPTY */) {
                        this.cells[this.model.toFlat(q, r)] = this.drawTile(canvas, q, r, e);
                    } else if (e.type != 0 /* OUT_OF_BOUNDS */) {
                        // just draw an empty tile anyway
                        this.drawTile(canvas, q, r, new Tile(1 /* EMPTY */, -1));
                    }
                }
            }

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(q, r);
                    if (e.type == 2 /* REGULAR */) {
                        this.cells[this.model.toFlat(q, r)] = this.drawTile(canvas, q, r, e);
                    }
                }
            }
        };

        HexView.prototype.getDOMElements = function () {
            return this.cells;
        };

        HexView.prototype.getDOMElement = function (q, r) {
            return Math.abs(q) <= this.model.gridr && Math.abs(r) <= this.model.gridr ? this.cells[this.model.toFlat(q, r)] : null;
        };
        return HexView;
    })();
    View.HexView = HexView;
})(View || (View = {}));
