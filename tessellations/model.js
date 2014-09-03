var Model;
(function (Model) {
    var Square = (function () {
        function Square(gridw, gridh, DefaultTile, OutOfBoundsTile) {
            this.outOfBoundsTile = OutOfBoundsTile;
            this.gridw = gridw;
            this.gridh = gridh;
            this.size = gridw * gridh;
            this.grid = [];
            for (var i = 0; i < this.gridw * this.gridh; i++) {
                this.grid.push(DefaultTile);
            }
        }
        Square.prototype.get = function (x, y) {
            return y >= 0 && x >= 0 && x < this.gridw && y < this.gridh ? this.grid[x + y * this.gridw] : this.outOfBoundsTile;
        };

        Square.prototype.getFlat = function (i) {
            return i >= 0 && i < this.gridw * this.gridh ? this.grid[i] : this.outOfBoundsTile;
        };

        Square.prototype.set = function (x, y, tile) {
            if (y >= 0 && x >= 0 && x < this.gridw && y < this.gridh) {
                this.grid[x + y * this.gridw] = tile;
            }
        };

        Square.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.gridw * this.gridh) {
                this.grid[i] = tile;
            }
        };
        return Square;
    })();
    Model.Square = Square;

    // TODO: implement me
    var Hex = (function () {
        function Hex() {
        }
        Hex.prototype.get = function (x, y) {
            return null;
        };
        Hex.prototype.getFlat = function (i) {
            return null;
        };
        Hex.prototype.set = function (x, y, tile) {
        };
        Hex.prototype.setFlat = function (i, tile) {
        };
        return Hex;
    })();
    Model.Hex = Hex;
})(Model || (Model = {}));
