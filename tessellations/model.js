var TileType;
(function (TileType) {
    TileType[TileType["OUT_OF_BOUNDS"] = 0] = "OUT_OF_BOUNDS";
    TileType[TileType["EMPTY"] = 1] = "EMPTY";
    TileType[TileType["REGULAR"] = 2] = "REGULAR";
    TileType[TileType["CONCRETE"] = 3] = "CONCRETE";
    TileType[TileType["DEACTIVATED"] = 4] = "DEACTIVATED";
    TileType[TileType["LAVA"] = 5] = "LAVA";
})(TileType || (TileType = {}));
;

var Tile = (function () {
    function Tile(t, v) {
        this.type = t;
        this.value = v;
    }
    return Tile;
})();

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

        Square.prototype.getTileArray = function () {
            return this.grid;
        };
        return Square;
    })();
    Model.Square = Square;

    var Hex = (function () {
        function Hex(gridr, DefaultTile, OutOfBoundsTile) {
            this.grid = [];
            this.gridr = gridr;
            this.outOfBoundsTile = OutOfBoundsTile;
            for (var r = -this.gridr; r <= this.gridr; r++) {
                for (var q = -this.gridr; q <= this.gridr; q++) {
                    if (Math.abs(r + q) > this.gridr)
                        this.grid.push(OutOfBoundsTile);
                    else
                        this.grid.push(DefaultTile);
                }
            }
        }
        // we need to make sure diameter is always odd...
        // in hex grid units
        Hex.prototype.diameter = function () {
            return (this.gridr * 2 + 1);
        };

        // q, r are relative to the center (IE: ( gridr, gridr ) in grid), convert it into the absolute index
        Hex.prototype.toFlat = function (q, r) {
            var x = r + this.gridr;
            var z = q + this.gridr;
            return x + z * (this.diameter());
        };

        Hex.prototype.get = function (q, r) {
            return Math.abs(q) <= this.gridr && Math.abs(r) <= this.gridr ? this.grid[this.toFlat(q, r)] : this.outOfBoundsTile;
        };

        Hex.prototype.getFlat = function (i) {
            return i >= 0 && i < this.grid.length ? this.grid[i] : this.outOfBoundsTile;
        };

        Hex.prototype.set = function (q, r, tile) {
            if (Math.abs(q) <= this.gridr && Math.abs(r) <= this.gridr) {
                this.grid[this.toFlat(q, r)] = tile;
            }
        };

        Hex.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.grid.length) {
                this.grid[i] = tile;
            }
        };

        Hex.prototype.getTileArray = function () {
            return this.grid;
        };
        return Hex;
    })();
    Model.Hex = Hex;
})(Model || (Model = {}));
